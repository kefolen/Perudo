# agents/mc_agent.py
import random
from sim.perudo import Action
from agents.baseline_agent import BaselineAgent

class MonteCarloAgent:
    def __init__(self, name='mc', n=200, rng=None):
        self.name = name
        self.N = n
        self.rng = rng or random.Random()
        self.rollout_policy_cls = BaselineAgent

    def select_action(self, obs):
        sim = obs.get('_simulator')
        current_bid = obs['current_bid']
        possible_actions = sim.legal_actions({'dice_counts': obs['dice_counts']}, current_bid, obs.get('palifico_restrict_face'))
        if current_bid is None:
            possible_actions = [a for a in possible_actions if a[0] != 'call']
        best = None
        best_val = -1e9
        for a in possible_actions:
            val = self.evaluate_action(obs, a)
            if val > best_val:
                best_val = val
                best = a
        return best

    def sample_determinization(self, obs):
        sim = obs.get('_simulator')
        DC = obs['dice_counts']
        my_idx = obs['player_idx']
        hands = [None] * sim.num_players
        hands[my_idx] = list(obs['my_hand'])
        for i in range(sim.num_players):
            if i == my_idx:
                continue
            if DC[i] <= 0:
                hands[i] = []
            else:
                hands[i] = [self.rng.randint(1, 6) for _ in range(DC[i])]
        return hands

    def evaluate_action(self, obs, action):
        total = 0.0
        for _ in range(self.N):
            hands = self.sample_determinization(obs)
            r = self.simulate_from_determinization(hands, obs, action)
            total += r
        return total / max(1, self.N)

    def simulate_from_determinization(self, full_hands, obs, first_action):
        sim = obs.get('_simulator')
        state = {'dice_counts': list(obs['dice_counts'])}
        hands = [list(h) for h in full_hands]
        current_player = obs['player_idx']
        current_bid = obs['current_bid']
        palifico_active = obs['palifico_active']
        palifico_restrict_face = obs.get('palifico_restrict_face')
        # apply first action
        if first_action[0] == 'bid':
            current_bid = (first_action[1], first_action[2])
            last_bid_maker = current_player
        elif first_action[0] == 'call':
            if current_bid is None:
                return 0.0
            true, cnt = sim.is_bid_true(hands, current_bid, ones_are_wild=(sim.ones_are_wild and not (palifico_active and state['dice_counts'][current_player]==1)))
            loser = current_player if true else (last_bid_maker if 'last_bid_maker' in locals() else sim.next_player_idx(current_player - 1, state['dice_counts']))
            state['dice_counts'][loser] = max(0, state['dice_counts'][loser] - 1)
        elif first_action[0] == 'exact':
            if current_bid is None:
                return 0.0
            true, cnt = sim.is_bid_true(hands, current_bid, ones_are_wild=(sim.ones_are_wild and not (palifico_active and state['dice_counts'][current_player]==1)))
            if cnt == current_bid[0]:
                for i in range(sim.num_players):
                    if i != current_player and state['dice_counts'][i] > 0:
                        state['dice_counts'][i] = max(0, state['dice_counts'][i] - 1)
            else:
                state['dice_counts'][current_player] = max(0, state['dice_counts'][current_player] - 1)
        # continue using baseline agents until game end (approximate)
        rollout_agents = [self.rollout_policy_cls(name=f"roll_{i}") for i in range(sim.num_players)]
        cur = sim.next_player_idx(current_player, state['dice_counts'])
        last_bid_maker = locals().get('last_bid_maker', None)
        while sum(1 for c in state['dice_counts'] if c > 0) > 1:
            agent = rollout_agents[cur]
            obs_local = {
                'player_idx': cur,
                'my_hand': list(hands[cur]),
                'dice_counts': list(state['dice_counts']),
                'current_bid': current_bid,
                'palifico_active': palifico_active,
                'palifico_restrict_face': palifico_restrict_face,
                '_simulator': sim
            }
            a = agent.select_action(obs_local)
            if a[0] == 'call':
                if current_bid is None:
                    cur = sim.next_player_idx(cur, state['dice_counts'])
                    continue
                true, cnt = sim.is_bid_true(hands, current_bid, ones_are_wild=(sim.ones_are_wild and not (palifico_active and state['dice_counts'][current_player]==1)))
                loser = cur if true else (last_bid_maker if last_bid_maker is not None else sim.next_player_idx(cur - 1, state['dice_counts']))
                state['dice_counts'][loser] = max(0, state['dice_counts'][loser] - 1)
                # regenerate hands
                for i in range(sim.num_players):
                    if state['dice_counts'][i] > 0:
                        hands[i] = [random.randint(1,6) for _ in range(state['dice_counts'][i])]
                    else:
                        hands[i] = []
                current_bid = None
                last_bid_maker = None
                cur = loser
                continue
            elif a[0] == 'exact':
                if current_bid is None:
                    cur = sim.next_player_idx(cur, state['dice_counts'])
                    continue
                true, cnt = sim.is_bid_true(hands, current_bid, ones_are_wild=(sim.ones_are_wild and not (palifico_active and state['dice_counts'][current_player]==1)))
                if cnt == current_bid[0]:
                    for i in range(sim.num_players):
                        if i != cur and state['dice_counts'][i] > 0:
                            state['dice_counts'][i] = max(0, state['dice_counts'][i] - 1)
                else:
                    state['dice_counts'][cur] = max(0, state['dice_counts'][cur] - 1)
                for i in range(sim.num_players):
                    if state['dice_counts'][i] > 0:
                        hands[i] = [random.randint(1,6) for _ in range(state['dice_counts'][i])]
                    else:
                        hands[i] = []
                current_bid = None
                last_bid_maker = None
                cur = sim.next_player_idx(cur, state['dice_counts'])
                continue
            elif a[0] == 'bid':
                current_bid = (a[1], a[2])
                last_bid_maker = cur
            cur = sim.next_player_idx(cur, state['dice_counts'])
        alive = [i for i, c in enumerate(state['dice_counts']) if c > 0]
        winner = alive[0] if len(alive) >= 1 else None
        return 1.0 if winner == obs['player_idx'] else 0.0