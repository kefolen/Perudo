import random
import math
import multiprocessing as mp
from sim.perudo import Action, binom_cdf_ge_fast
from agents.baseline_agent import BaselineAgent
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
                 history_memory_rounds=10, bayesian_sampling=False):
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
        
        # Initialize betting history parameters (Phase 5)
        self.betting_history_enabled = bool(betting_history_enabled)
        self.player_trust_enabled = bool(player_trust_enabled)
        self.trust_learning_rate = float(trust_learning_rate)
        self.history_memory_rounds = int(history_memory_rounds)
        self.bayesian_sampling = bool(bayesian_sampling)
        
        # Initialize parallel processing parameters using mixin
        self._initialize_parallel_parameters(enable_parallel, num_workers)

    def __del__(self):
        """Cleanup worker pool when agent is destroyed."""
        self._close_worker_pool()

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
            hands = self.sample_determinization(obs)
            return self.simulate_from_determinization(hands, obs, action)

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


    def select_action(self, obs, prune_k=12):
        """Top-level selection: optionally prune candidate actions (by simple prior),
           then evaluate each action and pick best. prune_k=number of bids to keep (not counting call/exact).
        """
        sim = obs['_simulator']
        current_bid = obs['current_bid']
        cand = sim.legal_actions({'dice_counts': obs['dice_counts']}, current_bid, obs.get('maputa_restrict_face'))
        if current_bid is None:
            cand = [a for a in cand if a[0] != 'call']

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

        for a in cand_pruned:
            val = self.evaluate_action(obs, a, best_so_far=best_val)
            if val > best_val:
                best_val = val
                best_action = a

        # fallback
        return best_action if best_action is not None else random.choice(cand_pruned)

    def simulate_from_determinization(self, full_hands, obs, first_action):
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
