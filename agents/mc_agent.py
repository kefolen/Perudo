import random
import math
import multiprocessing as mp
from sim.perudo import Action, binom_cdf_ge_fast
from agents.baseline_agent import BaselineAgent
from agents.random_agent import RandomAgent
from agents.mc_utils import (
    sample_weighted_dice, compute_enhanced_action_score,
    compute_control_variate_baseline, apply_variance_reduction,
    heuristic_win_prob, sample_bayesian_player_dice, compute_collective_plausibility
)
from agents.mc_parallel import ParallelProcessingMixin, worker_run_chunk


def _worker_run_chunk(agent, obs, action, chunk_size, seed_offset):
    """Module-level worker function for parallel chunk processing (backward compatibility)."""
    return worker_run_chunk(agent, obs, action, chunk_size, seed_offset)


class MonteCarloAgent(ParallelProcessingMixin):
    """
    Faster Monte-Carlo determinization agent with:
      - chunked simulations
      - precomputed binomial tails for p=1/6 and p=1/3
      - early stopping (per-action) via optional early-terminate by margin
      - minimal allocations in inner loop
      - simulates until game ends naturally (only one player remains)
      - weighted determinization sampling based on bidding history (Phase 1)
      - parallel processing support with configurable worker pools (Phase 2)
      - enhanced action pruning with opponent modeling (Phase 3)
      - variance reduction techniques using control variates (Phase 3)
    """

    def __init__(self, name='mc', n=200, rng=None,
                 chunk_size=8, early_stop_margin=0.1, weighted_sampling=False,
                 enable_parallel=False, num_workers=None, enhanced_pruning=False,
                 variance_reduction=False, betting_history_enabled=False,
                 player_trust_enabled=False, trust_learning_rate=0.1,
                 history_memory_rounds=10, bayesian_sampling=False,
                 call_eval_mode='rollout',
                 mixture_enabled=False, mixture_policies=None, mixture_weights=None,
                 mixture_min_weight_floor=0.0, mixture_depth_schedule=None,
                 confstop_enabled=False, confstop_delta=0.05, confstop_min_init=8,
                 confstop_batch_size=16, confstop_max_total_rollouts=None,
                 confstop_max_per_action=None, confstop_bound_b=1.0):
        self.name = name
        self.N = max(1, int(n))
        self.rng = rng or random.Random()
        self.chunk_size = max(1, int(chunk_size))
        self.rollout_policy_cls = BaselineAgent
        self._rollout_agents = None
        # early_stop_margin: if an action is behind current best by > margin (estimate), stop evaluating it
        self.early_stop_margin = float(early_stop_margin)
        self.weighted_sampling = bool(weighted_sampling)
        self.enhanced_pruning = bool(enhanced_pruning)
        self.variance_reduction = bool(variance_reduction)
        
        # Initialize betting history parameters
        self.betting_history_enabled = bool(betting_history_enabled)
        self.player_trust_enabled = bool(player_trust_enabled)
        self.trust_learning_rate = float(trust_learning_rate)
        self.history_memory_rounds = int(history_memory_rounds)
        self.bayesian_sampling = bool(bayesian_sampling)
        
        # Initialize parallel processing parameters using mixin
        self._initialize_parallel_parameters(enable_parallel, num_workers)
        # Analytical call evaluation mode: 'rollout' (default), 'analytic_leaf' or 'hybrid' (future)
        self.call_eval_mode = call_eval_mode

        # --- Mixture rollout configuration (Milestone 2) ---
        self.mixture_enabled = bool(mixture_enabled)
        # Policies list
        self.mixture_policies = list(mixture_policies) if mixture_policies is not None else ['baseline']
        # Normalize and floor weights
        self.mixture_min_weight_floor = float(mixture_min_weight_floor)
        self.mixture_depth_schedule = mixture_depth_schedule  # optional dict with thresholds/weights
        self.mixture_weights = self._normalize_with_floor(
            mixture_weights if mixture_weights is not None else [1.0] * len(self.mixture_policies),
            self.mixture_policies,
            self.mixture_min_weight_floor,
        )
        # Cached per-policy per-player rollout agents
        self._rollout_agents_by_policy = None

        # --- Confidence-based early stopping (Milestone 3) config ---
        self.confstop_enabled = bool(confstop_enabled)
        self.confstop_delta = float(confstop_delta)
        self.confstop_min_init = int(confstop_min_init)
        self.confstop_batch_size = int(confstop_batch_size)
        self.confstop_max_total_rollouts = None if confstop_max_total_rollouts is None else int(confstop_max_total_rollouts)
        self.confstop_max_per_action = None if confstop_max_per_action is None else int(confstop_max_per_action)
        self.confstop_bound_b = float(confstop_bound_b)

    def __del__(self):
        """Cleanup worker pool when agent is destroyed."""
        self._close_worker_pool()

    # --- Analytical call evaluation helpers (Milestone 1) ---
    def _compute_call_truth_probability(self, obs):
        """Compute P_true that the current bid is true given observation.
        Uses fast binomial tail lookup. Respects ones-are-wild and maputa rules.
        """
        sim = obs['_simulator']
        current_bid = obs.get('current_bid')
        if current_bid is None:
            return 0.0
        qty, face = current_bid
        my_hand = obs['my_hand']
        total_dice = sum(obs['dice_counts'])
        unknown = total_dice - len(my_hand)
        # own counts
        x_f = sum(1 for d in my_hand if d == face)
        x_1 = sum(1 for d in my_hand if d == 1)
        ones_wild_effective = sim.ones_are_wild and not obs.get('maputa_active', False)
        if face != 1 and ones_wild_effective:
            p = 1/3
            need = max(0, qty - (x_f + x_1))
        else:
            p = 1/6
            if face == 1:
                need = max(0, qty - x_1)
            else:
                need = max(0, qty - x_f)
        return binom_cdf_ge_fast(unknown, need, p) if unknown >= 0 else 0.0

    def _analytic_call_value(self, obs):
        """Return E[Δdice] = 1 - 2*P_true for calling current bid.
        Note: This value is on a different scale than terminal win probability.
        """
        P_true = self._compute_call_truth_probability(obs)
        return 1.0 - 2.0 * P_true

    class ActionStats:
        def __init__(self):
            self.n = 0
            self.mean = 0.0
            self.M2 = 0.0  # Sum of squares of differences from the current mean
        def update(self, x):
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.M2 += delta * delta2
        @property
        def var_unbiased(self):
            return self.M2 / (self.n - 1) if self.n > 1 else 0.0
        def radius(self, delta, b):
            if self.n <= 0:
                return float('inf')
            s2 = self.var_unbiased
            # Empirical-Bernstein radius
            term1 = math.sqrt(max(0.0, 2.0 * s2 * math.log(3.0 / max(1e-12, delta)) / self.n))
            term2 = 3.0 * b * math.log(3.0 / max(1e-12, delta)) / max(1, self.n - 1 + 1)
            # The denominator for term2 often uses n-1 or n; we keep n for stability
            term2 = 3.0 * b * math.log(3.0 / max(1e-12, delta)) / self.n
            return term1 + term2

    def _eb_radius(self, n, var_unbiased, delta=None, b=None):
        """Helper to compute Empirical-Bernstein radius given stats."""
        if n <= 0:
            return float('inf')
        if delta is None:
            delta = self.confstop_delta
        if b is None:
            b = self.confstop_bound_b
        term1 = math.sqrt(max(0.0, 2.0 * var_unbiased * math.log(3.0 / max(1e-12, delta)) / n))
        term2 = 3.0 * b * math.log(3.0 / max(1e-12, delta)) / n
        return term1 + term2

    def _normalize_with_floor(self, weights, policies, floor):
        # Ensure length matches
        if weights is None or len(weights) != len(policies):
            weights = [1.0] * len(policies)
        # Apply floor and renormalize
        w = [max(float(floor), float(x)) for x in weights]
        s = sum(w)
        if s <= 0:
            w = [1.0 / len(w)] * len(w)
        else:
            w = [x / s for x in w]
        return w

    def _policy_factory(self, policy_id):
        pid = str(policy_id).lower()
        if pid == 'baseline':
            return lambda name: BaselineAgent(name=name)
        if pid == 'baseline_conservative':
            return lambda name: BaselineAgent(name=name, threshold_call=0.7)
        if pid == 'baseline_aggressive':
            return lambda name: BaselineAgent(name=name, threshold_call=0.3)
        if pid == 'random':
            # Seeded random agent is not necessary for rollouts; keep simple
            return lambda name: RandomAgent(name=name)
        # default fallback
        return lambda name: BaselineAgent(name=name)

    def _ensure_rollout_agents_by_policy(self, sim):
        if self._rollout_agents_by_policy is None:
            self._rollout_agents_by_policy = {}
            for pid in self.mixture_policies:
                factory = self._policy_factory(pid)
                self._rollout_agents_by_policy[pid] = [factory(name=f"{pid}_{i}") for i in range(sim.num_players)]

    def _get_depth_weights(self, depth):
        # Depth schedule optional; default to static weights
        return self.mixture_weights

    def _sample_policy_id(self, depth=0, rng=None):
        rng = rng or self.rng
        if not self.mixture_enabled:
            return 'baseline'
        # If only one policy, avoid consuming RNG to preserve stream parity
        if len(self.mixture_policies) == 1:
            return self.mixture_policies[0]
        weights = self._get_depth_weights(depth)
        # Python's random.choices
        return rng.choices(self.mixture_policies, weights=weights, k=1)[0]

    # sample determinization (fast)
    def sample_determinization(self, obs):
        sim = obs['_simulator']
        DC = obs['dice_counts']
        my_idx = obs['player_idx']
        # reuse rng locally
        randint = self.rng.randint
        hands = [None] * sim.num_players
        hands[my_idx] = obs['my_hand']  # pass existing list (do NOT copy)
        
        for i in range(sim.num_players):
            if i == my_idx:
                continue
            if DC[i] <= 0:
                hands[i] = []
            else:
                # Use Bayesian sampling if enabled and history available
                if self.bayesian_sampling and obs.get('betting_history') is not None:
                    hands[i] = sample_bayesian_player_dice(self, obs, i, DC[i])
                elif self.weighted_sampling:
                    hands[i] = sample_weighted_dice(self, obs, DC[i])
                else:
                    # faster generation: use list comprehension with local randint
                    hands[i] = [randint(1, 6) for _ in range(DC[i])]
        return hands






    def evaluate_action(self, obs, action, best_so_far=None):
        """Evaluate `action` by running self.N simulations in chunks.
           best_so_far: mean of current best action (optional) for early stopping.
        """
        # Ensure rollout agents are initialized
        sim = obs['_simulator']
        if self._rollout_agents is None or len(self._rollout_agents) != sim.num_players:
            self._rollout_agents = [self.rollout_policy_cls(name=f"roll_{i}") for i in range(sim.num_players)]
        # Ensure mixture caches if enabled
        if self.mixture_enabled:
            self._ensure_rollout_agents_by_policy(sim)
        
        if self.enable_parallel and self.num_workers > 1:
            return self._evaluate_action_parallel(obs, action, best_so_far)
        else:
            return self._evaluate_action_sequential(obs, action, best_so_far)

    def _evaluate_action_sequential(self, obs, action, best_so_far=None):
        """Sequential evaluation (original implementation)."""
        # get legal actions count to scale sims if needed (already done outside if desired)
        sims = self.N
        chunk = self.chunk_size
        full_chunks = sims // chunk
        rem = sims % chunk

        total = 0.0
        total_sq = 0.0  # for variance if needed
        runs_done = 0
        
        # For variance reduction, collect individual results
        simulation_results = [] if self.variance_reduction else None

        # localize frequently used objects to speed up loop
        sim = obs['_simulator']
        player_idx = obs['player_idx']

        # early stop thresholds derived from best_so_far
        early_margin = self.early_stop_margin
        best_mean = None
        if best_so_far is not None:
            best_mean = best_so_far

        def run_one_sim():
            # Milestone 1: analytical evaluation for 'call' action (experimental)
            if action[0] == 'call' and getattr(self, 'call_eval_mode', 'rollout') == 'analytic_leaf':
                return self._analytic_call_value(obs)
            hands = self.sample_determinization(obs)
            rollout_agents = None
            if self.mixture_enabled:
                # Sample a policy id for this rollout (static mixture)
                pid = self._sample_policy_id(depth=0, rng=self.rng)
                rollout_agents = self._rollout_agents_by_policy.get(pid)
            return self.simulate_from_determinization(hands, obs, action, rollout_agents=rollout_agents)

        # run full chunks
        for _ in range(full_chunks):
            # chunked loop (reduce overhead)
            for _c in range(chunk):
                r = run_one_sim()
                total += r
                total_sq += r * r
                runs_done += 1
                
                # Store result for variance reduction if enabled
                if simulation_results is not None:
                    simulation_results.append(r)
                    
            # after each chunk check early stop (if provided)
            if best_mean is not None and runs_done > 8:
                mean = total / runs_done
                # simple early check: if mean + margin < best_mean -> stop
                if mean + early_margin < best_mean:
                    break

        # remainder
        for _ in range(rem):
            r = run_one_sim()
            total += r
            total_sq += r * r
            runs_done += 1
            
            # Store result for variance reduction if enabled
            if simulation_results is not None:
                simulation_results.append(r)
                
            if best_mean is not None and runs_done > 8:
                mean = total / runs_done
                if mean + early_margin < best_mean:
                    break

        if runs_done == 0:
            return 0.0
            
        # Apply variance reduction if enabled
        if self.variance_reduction and simulation_results:
            return apply_variance_reduction(simulation_results, obs, action)
        else:
            return total / runs_done

    def _run_single_rollout_value(self, obs, action):
        """Run a single rollout for the given action and return its value.
        Respects analytic call mode and mixture rollout settings.
        """
        # Analytical short-circuit for 'call' analytic_leaf mode
        if action[0] == 'call' and getattr(self, 'call_eval_mode', 'rollout') == 'analytic_leaf':
            return self._analytic_call_value(obs)
        hands = self.sample_determinization(obs)
        rollout_agents = None
        if getattr(self, 'mixture_enabled', False):
            pid = self._sample_policy_id(depth=0, rng=self.rng)
            rollout_agents = None if self._rollout_agents_by_policy is None else self._rollout_agents_by_policy.get(pid)
        return self.simulate_from_determinization(hands, obs, action, rollout_agents=rollout_agents)

    def _run_k_rollouts_for_action(self, obs, action, k):
        results = []
        for _ in range(k):
            results.append(self._run_single_rollout_value(obs, action))
        return results

    def _select_action_confstop(self, obs, cand_actions):
        """Empirical-Bernstein LUCB best-arm identification among cand_actions.
        Returns the selected action.
        """
        if not cand_actions:
            return None
        if len(cand_actions) == 1:
            return cand_actions[0]

        # Ensure rollout agents initialized (baseline and mixture caches)
        sim = obs.get('_simulator')
        if self._rollout_agents is None and sim is not None:
            self._rollout_agents = [self.rollout_policy_cls(name=f"roll_{i}") for i in range(sim.num_players)]
        if getattr(self, 'mixture_enabled', False) and sim is not None:
            self._ensure_rollout_agents_by_policy(sim)

        # Stats per action
        stats = {a: MonteCarloAgent.ActionStats() for a in cand_actions}
        total_budget = self.confstop_max_total_rollouts if self.confstop_max_total_rollouts is not None else float('inf')
        per_action_cap = self.confstop_max_per_action if self.confstop_max_per_action is not None else float('inf')
        batch = max(1, int(self.confstop_batch_size))
        min_init = max(1, int(self.confstop_min_init))
        delta = float(self.confstop_delta)
        b = float(self.confstop_bound_b)

        total_used = 0
        used_per_action = {a: 0 for a in cand_actions}

        # Warm start: min_init per action
        for a in cand_actions:
            k = min(min_init, int(per_action_cap - used_per_action[a]), int(total_budget - total_used))
            if k <= 0:
                continue
            vals = self._run_k_rollouts_for_action(obs, a, k)
            total_used += k
            used_per_action[a] += k
            for v in vals:
                stats[a].update(v)

        # Main LUCB loop
        while total_used < total_budget:
            # Compute bounds
            means = {a: stats[a].mean for a in cand_actions}
            radii = {a: stats[a].radius(delta, b) for a in cand_actions}
            # Identify current best by mean
            a_star = max(cand_actions, key=lambda a: means[a])
            l_star = means[a_star] - radii[a_star]
            # Challenger: highest UCB among others
            challenger = None
            max_ucb = -float('inf')
            for a in cand_actions:
                if a == a_star:
                    continue
                u = means[a] + radii[a]
                if u > max_ucb:
                    max_ucb = u
                    challenger = a
            if challenger is None:
                break
            # Stopping condition: U(challenger) <= L(star)
            if max_ucb <= l_star:
                return a_star

            # Determine pulls for star and challenger
            def pulls_for(a):
                rem_total = int(total_budget - total_used)
                rem_action = int(per_action_cap - used_per_action[a])
                return max(0, min(batch, rem_total, rem_action))

            k1 = pulls_for(a_star)
            k2 = pulls_for(challenger)
            if k1 <= 0 and k2 <= 0:
                break
            if k1 > 0:
                s1 = self._run_k_rollouts_for_action(obs, a_star, k1)
                total_used += k1
                used_per_action[a_star] += k1
                for _ in range(k1):
                    stats[a_star].update(s1 / k1)
            if k2 > 0:
                s2 = self._run_k_rollouts_for_action(obs, challenger, k2)
                total_used += k2
                used_per_action[challenger] += k2
                for _ in range(k2):
                    stats[challenger].update(s2 / k2)

        # Fallback: choose best empirical mean
        return max(cand_actions, key=lambda a: stats[a].mean)

    def select_action(self, obs, prune_k=12):
        """Top-level selection: optionally prune candidate actions (by simple prior),
           then evaluate each action and pick best. prune_k=number of bids to keep (not counting call/exact).
        """
        sim = obs['_simulator']
        current_bid = obs['current_bid']
        cand = sim.legal_actions({'dice_counts': obs['dice_counts']}, current_bid, obs.get('maputa_restrict_face'))
        if current_bid is None:
            cand = [a for a in cand if (a[0] != 'call' and a[0] != 'exact')]

        # Fast prior: compute binomial plausibility for each bid and sort
        bids = []
        others = []
        TD = sum(obs['dice_counts'])
        for a in cand:
            if a[0] == 'bid':
                qty, face = a[1], a[2]
                # compute quick prior P(at least qty) using fast binom lookup
                # approximate k = my_count for face
                my_hand = obs['my_hand']
                k = sum(1 for d in my_hand if d == face)
                n_other = TD - len(my_hand)
                if face != 1 and obs.get('maputa_active', False) is False and sim.ones_are_wild:
                    p = 1 / 3
                else:
                    p = 1 / 6
                need = max(0, qty - k)
                prior = binom_cdf_ge_fast(n_other, need, p) if n_other >= 0 else 0.0
                
                # Use enhanced scoring if enabled
                if self.enhanced_pruning:
                    score = compute_enhanced_action_score(obs, a, prior)
                else:
                    score = prior
                
                bids.append((score, a))
            else:
                others.append((1.0, a))
        # keep top-k bids by score (enhanced or standard prior)
        bids.sort(reverse=True, key=lambda x: x[0])
        kept_bids = [b for (_, b) in bids[:prune_k]]
        cand_pruned = kept_bids + [a for (_, a) in others]

        # evaluate actions; use best mean as early-stop reference
        best_action = None
        best_val = -1.0

        # ensure rollout agents cached
        if self._rollout_agents is None or len(self._rollout_agents) != sim.num_players:
            self._rollout_agents = [self.rollout_policy_cls(name=f"roll_{i}") for i in range(sim.num_players)]

        # If confidence-based early stopping is enabled, use LUCB controller
        if getattr(self, 'confstop_enabled', False):
            chosen = self._select_action_confstop(obs, cand_pruned)
            if chosen is not None:
                return chosen

        for a in cand_pruned:
            val = self.evaluate_action(obs, a, best_so_far=best_val)
            if val > best_val:
                best_val = val
                best_action = a

        # fallback
        return best_action if best_action is not None else random.choice(cand_pruned)

    def simulate_from_determinization(self, full_hands, obs, first_action, rollout_agents=None):
        """
        Optimized simulate:
          - local variables for performance
          - simulates until game ends naturally (only one player remains)
        """
        sim = obs['_simulator']
        # shallow copies only where needed
        dice_counts = list(obs['dice_counts'])
        hands = [list(h) for h in full_hands]
        cur = obs['player_idx']
        current_bid = obs['current_bid']
        maputa_active = obs.get('maputa_active', False)
        maputa_restrict_face = obs.get('maputa_restrict_face', None)
        player_idx = obs['player_idx']

        # compute alive_count and maintain it
        alive_count = sum(1 for c in dice_counts if c > 0)

        # helper locals
        nxt = sim.next_player_idx
        is_bid_true = sim.is_bid_true
        ones_are_wild = sim.ones_are_wild
        randint = self.rng.randint

        # set current_bid_maker (approx: previous player if there was a bid)
        current_bid_maker = None
        if current_bid is not None:
            # assume previous alive player was last bidder
            # find previous alive
            n = len(dice_counts)
            for off in range(1, n + 1):
                idx = (cur - off) % n
                if dice_counts[idx] > 0:
                    current_bid_maker = idx
                    break

        # apply first_action
        first_bid_by = None
        if first_action[0] == 'bid':
            first_bid_by = cur
            if maputa_active:
                maputa_restrict_face = first_action[2]
            current_bid = (first_action[1], first_action[2])
            current_bid_maker = cur
            cur = nxt(cur, dice_counts)
        else:
            if first_action[0] == 'call':
                if current_bid is None:
                    return 0.0
                bid_true, cnt = is_bid_true(hands, current_bid, ones_are_wild=(ones_are_wild and not maputa_active))
                loser = cur if bid_true else current_bid_maker
            else:  # exact
                if current_bid is None:
                    return 0.0
                bid_true, cnt = is_bid_true(hands, current_bid, ones_are_wild=(ones_are_wild and not maputa_active))
                if cnt == current_bid[0]:
                    dice_counts[cur] += 1
                    loser = current_bid_maker
                else:

                    loser = cur

            dice_counts[loser] = max(0, dice_counts[loser] - 1)
            if dice_counts[loser] == 0:
                alive_count -= 1
            # reset round quickly (regenerate hands for alive players)
            for i in range(len(dice_counts)):
                if dice_counts[i] > 0:
                    hands[i] = [randint(1, 6) for _ in range(dice_counts[i])]
                else:
                    hands[i] = []
            current_bid = None
            current_bid_maker = None
            maputa_restrict_face = None
            maputa_active = sim.use_maputa and (dice_counts[loser] == 1)
            cur = loser

        # Simulate until game ends naturally (only one player remains)
        while alive_count > 1:
            # pick rollout agent for cur (they use obs only)
            # build minimal obs_local (reuse frequently)
            # Note: baseline agent expects lists; do not copy hands[cur] if agent doesn't mutate
            obs_local = {
                'player_idx': cur,
                'my_hand': hands[cur],
                'dice_counts': dice_counts,
                'current_bid': current_bid,
                'maputa_active': maputa_active,
                'maputa_restrict_face': maputa_restrict_face,
                '_simulator': sim
            }
            # Choose rollout agent depending on mixture selection
            if rollout_agents is not None:
                a = rollout_agents[cur].select_action(obs_local)
            else:
                a = self._rollout_agents[cur].select_action(obs_local)

            # handle action
            if Action.is_bid(a):
                if maputa_restrict_face is not None and a[2] != maputa_restrict_face:
                    # invalid -> loser is cur
                    loser = cur

                else:
                    current_bid = (a[1], a[2])
                    current_bid_maker = cur
                    if first_bid_by is None:
                        first_bid_by = cur
                        if maputa_active:
                            maputa_restrict_face = a[2]
                    cur = nxt(cur, dice_counts)
                    continue
            elif a[0] == 'call':
                if current_bid is None:
                    # invalid call => penalize caller
                    loser = cur

                else:
                    bid_true, cnt = is_bid_true(hands, current_bid,
                                                ones_are_wild=(ones_are_wild and not maputa_active))
                    loser = cur if bid_true else current_bid_maker
            else:  # exact
                if current_bid is None:
                    loser = cur
                else:
                    bid_true, cnt = is_bid_true(hands, current_bid,
                                                ones_are_wild=(ones_are_wild and not maputa_active))
                    if cnt == current_bid[0]:
                        dice_counts[cur] += 1
                        loser = current_bid_maker
                    else:
                        loser = cur

            dice_counts[loser] = max(0, dice_counts[loser] - 1)
            if dice_counts[loser] == 0:
                alive_count -= 1
            # reset round quickly (regenerate hands for alive players)
            for i in range(len(dice_counts)):
                if dice_counts[i] > 0:
                    hands[i] = [randint(1, 6) for _ in range(dice_counts[i])]
                else:
                    hands[i] = []
            current_bid = None
            current_bid_maker = None
            maputa_restrict_face = None
            maputa_active = sim.use_maputa and (dice_counts[loser] == 1)
            cur = loser

        # Game completed naturally
        if alive_count <= 0:
            return 0.0
        if alive_count == 1:
            # determine winner
            winner = None
            for i, c in enumerate(dice_counts):
                if c > 0:
                    winner = i
                    break
            return 1.0 if winner == player_idx else 0.0
        
        # This should not happen since we loop until alive_count <= 1
        return 0.0
