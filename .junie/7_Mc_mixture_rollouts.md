1) Rollout policy as a mixture (not a fixed baseline)

Goal
- Increase robustness and reduce rollout bias by sampling actions from a mixture of multiple rollout policies rather than a single fixed baseline.

Design
- Define a set of rollout policies P = {π1, π2, …, πK}. Examples:
  - Heuristic baseline policy
  - Risk-aware policy (conservative)
  - Aggressive policy (higher-variance bids)
  - History-aware policy (uses simple counts from betting history)
  - Random/uniform policy (ensures exploration)
- Sample a policy per rollout according to mixture weights w ∈ ΔK (weights sum to 1).

Core behaviors
- Static mixture: Use fixed weights w at all depths (simple and robust).
- Depth-dependent mixture: Use weight schedules w(d) that shift from exploratory near the root to more deterministic deeper in the tree.
- Context-gated mixture (optional): Allow features to reweight the mixture at rollout time:
  - Features could include remaining dice, last bid gap, number of players, and whether ones are active.
  - Implement as softmax over a linear score of features per policy or simple rules (e.g., more conservative when few dice remain).

Adaptive weighting (optional, off by default)
- Online bandit adaptation per decision:
  - Track running mean rollout return per policy.
  - Update weights via softmax over policy scores with temperature τ.
  - Add a Dirichlet prior α for stability to avoid collapsing to one policy too quickly.
- Safeguards:
  - Minimum floor weight ε for all policies.
  - Reset or decay policy scores between decisions to limit drift.

Configuration
- mixture.enabled: bool
- mixture.policies: list of policy ids
- mixture.weights: list[float], same length as policies
- mixture.depth_schedule: {depth_thresholds: [int], weights: list[list[float]]} (optional)
- mixture.adaptive.enabled: bool
- mixture.adaptive.temperature: float
- mixture.adaptive.prior_alpha: float or list[float]
- mixture.min_weight_floor: float
- mixture.root_dirichlet: {alpha: float, epsilon: float} (optional, if injecting exploration noise at root)

Parallelism
- Choose the policy at the worker before each rollout using a stateless RNG seeded per rollout to maintain reproducibility across workers.
- Aggregate per-policy return statistics centrally if adaptive weighting is enabled.

2) Confidence-based early stopping (replace fixed margin)

Goal
- Replace fixed-margin early stop with a statistically sound “best action identification” criterion using confidence intervals, reducing wasted simulations while maintaining decision quality.

Design
- Use Empirical-Bernstein LUCB-style racing:
  - Maintain for each candidate action a: count n_a, mean μ_a, unbiased variance estimate s2_a.
  - Confidence radius β_a(n) = sqrt(2 s2_a ln(3/δ) / n_a) + 3b ln(3/δ)/n_a
    - δ: failure probability (e.g., 0.05)
    - b: bounded range of returns; if returns are clipped to [−1, 1], b = 1.
- Stopping rule
  - Let a* = argmax_a μ_a.
  - If μ_{a*} − β_{a*} > max_{a≠a*} (μ_a + β_a), stop and select a*.
  - Also stop if any hard budget constraints are reached.
- Sampling schedule
  - Pull the action with the largest upper confidence bound, or use LUCB pulls: repeatedly sample a* and the best competitor a’ = argmax_{a≠a*} (μ_a + β_a).
  - Batch rollouts per action to fit vectorized/parallel execution.

Return normalization
- If rollout returns are not naturally in [−1, 1], apply clipping or standardization (keep it consistent across actions) to keep b small and intervals tight.

Edge cases and guards
- Minimum initial samples per action: n_min_init (e.g., 5–10) before applying early stopping.
- Max samples per action: n_max to prevent outliers from consuming all budget.
- Tie-breaking: prefer higher μ_a; fallback to deterministic action ordering.

Configuration
- confstop.enabled: bool
- confstop.delta: float (e.g., 0.05)
- confstop.min_init: int
- confstop.batch_size: int
- confstop.max_total_rollouts: int
- confstop.max_per_action: int
- confstop.bound_b: float or auto (if returns are clipped)

Parallel aggregation
- Workers return (n, sum, sumsq) per action; coordinator computes μ, s2, and β.
- Use stable Welford updates to avoid numerical issues.

3) Analytical “call” evaluation to reduce variance

Goal
- Replace rollout-based evaluation for “call” with a closed-form probability, reducing variance and cost when evaluating a challenge.

Model
- Let the active bid be “q of face f”.
- Total dice in play: N. Player’s known dice: x_f of face f, x_1 of ones.
- Unknown dice among others: M = N − own_dice_count.
- Probability p that an unknown die contributes to count:
  - If f ≠ 1: p = 1/6 (face f) + 1/6 (ones are wild) = 1/3.
  - If f = 1: p = 1/6 (ones only).
- Required successes from unknown dice:
  - If f ≠ 1: r = max(0, q − (x_f + x_1)).
  - If f = 1: r = max(0, q − x_1).
- Probability that the bid is true:
  - P_true = sum_{k=r..M} C(M, k) p^k (1−p)^{M−k}.

Evaluation
- Immediate expected utility of calling:
  - If the bid is true, caller loses a die (−1); if false, bidder loses a die (+1).
  - E[Δdice] = (+1)·(1 − P_true) + (−1)·P_true = 1 − 2·P_true.
- Integration options:
  1) One-step analytic leaf (fastest, lowest variance):
     - Use E[Δdice] as the rollout return for the “call” action directly.
  2) Hybrid branching (slightly higher cost, still low variance):
     - Evaluate two successor branches deterministically with weights:
       - With prob P_true: next state where caller loses a die.
       - With prob 1 − P_true: next state where bidder loses a die.
     - Either:
       - Stop at depth 1 using a static value function for these states, or
       - Continue MC rollouts from each successor weighted by these probabilities.

Performance considerations
- Precompute or cache binomial CDFs for given (M, p, r) to avoid repeated summation.
- Use stable log-space computations for large M:
  - Compute tail probabilities via log-CDF or normal/Poisson approximations when appropriate.
- Extendable to “exact” calls (if supported):
  - Replace tail with equality probability P_exact = C(M, r) p^r (1−p)^{M−r}.
  - Adjust payoff logic accordingly.

Configuration
- call_eval.mode: enum {analytic_leaf, hybrid, rollout}
- call_eval.cache_enabled: bool
- call_eval.large_M_approx: enum {auto, normal, poisson, none}
- call_eval.use_logspace: bool

End-to-end decision flow

- Candidate action generation.
- For each action a:
  - If a is “call” and call_eval.mode != rollout:
    - Compute P_true analytically and score per chosen mode.
  - Else:
    - Use mixture rollout policy to simulate returns.
- Use confidence-based early stopping controller to adaptively allocate rollouts and stop when confident.
- Select action and proceed.

Pseudocode (high-level)

```python
def choose_action(state, actions, budget):
    stats = init_stats(actions)
    init_round(stats, conf=confstop)

    while not confstop.should_stop(stats) and not budget.exhausted():
        a_star = stats.argmax_mean()
        a_comp = stats.argmax_ucb(exclude=a_star)

        for a in [a_star, a_comp]:
            if a.is_call() and call_eval.mode != "rollout":
                reward = analytic_call_reward(state, a, call_eval)
                stats.update(a, reward, n=1)  # treat as a single exact sample
            else:
                n_batch = min(confstop.batch_size, budget.remaining())
                rewards = rollout_batch(state, a, n_batch, mixture_cfg)
                stats.update(a, rewards)

        budget.consume(work_done=...)

    return stats.best_action()

def rollout_batch(state, action, n, mixture_cfg):
    rewards = []
    for i in range(n):
        pi = sample_policy_from_mixture(mixture_cfg, depth=0, context=state.features())
        rewards.append(simulate_with_policy(state, action, pi))
    return rewards

def analytic_call_reward(state, call_action, cfg):
    q, f = call_action.bid
    N = state.total_dice
    x_f, x_1 = state.own_counts(face=f), state.own_counts(face=1)
    M = N - state.own_dice_count

    p = (1/3) if f != 1 else (1/6)
    r = max(0, q - (x_f + (x_1 if f != 1 else 0)))

    P_true = binomial_tail(M, p, r, cfg)
    # Option 1: analytic leaf (default)
    return 1 - 2 * P_true
```


Telemetry and controls
- Log:
  - Chosen rollout policy counts and per-policy average returns.
  - Early stopping diagnostics: μ, s2, β per action; number of rollouts saved vs. max budget.
  - Analytical call stats: M, p, r, P_true, mode used.
- Switches:
  - Feature flags to toggle each feature independently.
  - Safeguards to fall back to current baseline behavior if instability is detected.

Default parameter suggestions
- mixture.enabled: true
- mixture.weights (K=3 example): [0.5, 0.3, 0.2]
- mixture.min_weight_floor: 0.05
- confstop.enabled: true
- confstop.delta: 0.05
- confstop.min_init: 8
- confstop.batch_size: 16
- confstop.max_total_rollouts: keep current global budget
- call_eval.mode: analytic_leaf
- call_eval.cache_enabled: true
- call_eval.large_M_approx: auto
- Return clipping for confidence bounds: [-1, 1]

Risk and mitigation
- Mixture instability: enforce weight floors and optional temperature decay.
- Confidence miscalibration: start with conservative δ and minimum samples per action; monitor regret.
- Analytical errors for large M: use log-space and approximations; add unit-checked math utilities.

Milestones
- M1: Analytical call evaluation (leaf mode) + unit math checks and caching.
- M2: Mixture rollout scaffolding with static weights and depth schedule.
- M3: Confidence-based early stopping (Empirical-Bernstein LUCB) with parallel aggregation.
- M4: Optional adaptive mixture + telemetry and tuning.
