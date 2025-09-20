# Perudo (modular) — project template

This canvas contains a **modular version** of the single-file Perudo template. It includes recommended project layout, runnable code for each module, `requirements.txt`, and examples how to run tournaments. Save files from the sections below into separate files exactly as named.

---

## Project structure

```
perudo_project/
├── sim/
│   └── perudo.py            # game simulator (Perudo rules)
├── agents/
│   ├── __init__.py
│   ├── random_agent.py
│   ├── baseline_agent.py
│   └── mc_agent.py
├── eval/
│   └── tournament.py       # CLI to run matches/tournaments
├── requirements.txt
└── README.md
```

---

## sim/perudo.py

```python
# sim/perudo.py
import random
from collections import Counter
import math
from functools import lru_cache

# small helpers
@lru_cache(maxsize=1024)
def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0

@lru_cache(maxsize=1024)
def binom_cdf_ge(n, k, p):
    s = 0.0
    for j in range(k, n + 1):
        s += comb(n, j) * (p ** j) * ((1 - p) ** (n - j))
    return s


class Action:
    @staticmethod
    def bid(qty, face):
        return ("bid", int(qty), int(face))

    @staticmethod
    def call():
        return ("call",)

    @staticmethod
    def exact():
        return ("exact",)

    @staticmethod
    def is_bid(a):
        return a[0] == 'bid'

    @staticmethod
    def qty(a):
        return a[1] if Action.is_bid(a) else None

    @staticmethod
    def face(a):
        return a[2] if Action.is_bid(a) else None

    @staticmethod
    def to_str(a):
        if a[0] == 'bid':
            return f"bid {a[1]}x{a[2]}"
        return a[0]


class PerudoSimulator:
    def __init__(self, num_players=3, start_dice=5, ones_are_wild=True, use_palifico=True, use_exact=True, seed=None):
        self.num_players = num_players
        self.start_dice = start_dice
        self.ones_are_wild = ones_are_wild
        self.use_palifico = use_palifico
        self.use_exact = use_exact
        self.rng = random.Random(seed)

    def new_game(self):
        return {'dice_counts': [self.start_dice] * self.num_players}

    def roll_hands(self, state, full_hands=None):
        hands = []
        for i, dc in enumerate(state['dice_counts']):
            if dc <= 0:
                hands.append([])
            elif full_hands is not None and full_hands[i] is not None:
                hands.append(list(full_hands[i]))
            else:
                hands.append([self.rng.randint(1, 6) for _ in range(dc)])
        return hands

    def total_dice(self, state):
        return sum(state['dice_counts'])

    def legal_bids_after(self, current_bid, cap_qty):
        if current_bid is None:
            min_q, min_f = 1, 1
        else:
            min_q, min_f = current_bid
        for q in range(min_q, cap_qty + 1):
            for f in range(1, 7):
                if current_bid is not None:
                    if q == min_q and f <= min_f:
                        continue
                yield (q, f)

    def legal_actions(self, state, current_bid, palifico_restrict_face=None):
        actions = []
        TD = self.total_dice(state)
        if TD <= 0:
            return [Action.call()]
        for q, f in self.legal_bids_after(current_bid, TD):
            if palifico_restrict_face is not None and f != palifico_restrict_face:
                continue
            actions.append(Action.bid(q, f))
        actions.append(Action.call())
        if self.use_exact:
            actions.append(Action.exact())
        return actions

    def is_bid_true(self, hands, bid, ones_are_wild=True):
        qty, face = bid
        cnt = 0
        for hand in hands:
            cnt += sum(1 for d in hand if d == face)
            if ones_are_wild and face != 1:
                cnt += sum(1 for d in hand if d == 1)
        return cnt >= qty, cnt

    def next_player_idx(self, current_idx, dice_counts):
        n = len(dice_counts)
        for offset in range(1, n + 1):
            idx = (current_idx + offset) % n
            if dice_counts[idx] > 0:
                return idx
        return current_idx

    def play_game(self, agents, verbose=False):
        state = self.new_game()
        starting_player = 0
        while sum(1 for c in state['dice_counts'] if c > 0) > 1:
            alive = [i for i, c in enumerate(state['dice_counts']) if c > 0]
            if starting_player not in alive:
                starting_player = alive[0]
            palifico_active = self.use_palifico and (state['dice_counts'][starting_player] == 1)
            palifico_restrict_face = None
            hands = self.roll_hands(state)
            current_bid = None
            current_bid_maker = None
            current_player = starting_player
            first_bid_by = None
            while True:
                obs = {
                    'player_idx': current_player,
                    'my_hand': list(hands[current_player]),
                    'dice_counts': list(state['dice_counts']),
                    'current_bid': current_bid,
                    'history': [],
                    'palifico_active': palifico_active,
                    'palifico_restrict_face': palifico_restrict_face,
                    '_simulator': self,
                }
                action = agents[current_player].select_action(obs)
                if Action.is_bid(action) and first_bid_by is None:
                    first_bid_by = current_player
                    if palifico_active:
                        palifico_restrict_face = Action.face(action)
                if action[0] == 'call':
                    if current_bid is None:
                        # invalid
                        loser = current_player
                    else:
                        true, cnt = self.is_bid_true(hands, current_bid, ones_are_wild=(self.ones_are_wild and not (palifico_active and state['dice_counts'][starting_player]==1)))
                        if true:
                            loser = current_player
                        else:
                            loser = current_bid_maker
                    state['dice_counts'][loser] = max(0, state['dice_counts'][loser] - 1)
                    starting_player = loser
                    break
                elif action[0] == 'exact':
                    if current_bid is None:
                        state['dice_counts'][current_player] = max(0, state['dice_counts'][current_player] - 1)
                        starting_player = current_player
                        break
                    true, cnt = self.is_bid_true(hands, current_bid, ones_are_wild=(self.ones_are_wild and not (palifico_active and state['dice_counts'][starting_player]==1)))
                    if cnt == current_bid[0]:
                        for i in range(self.num_players):
                            if i != current_player and state['dice_counts'][i] > 0:
                                state['dice_counts'][i] = max(0, state['dice_counts'][i] - 1)
                        starting_player = current_player
                    else:
                        state['dice_counts'][current_player] = max(0, state['dice_counts'][current_player] - 1)
                        starting_player = current_player
                    break
                elif action[0] == 'bid':
                    qty, face = action[1], action[2]
                    if palifico_restrict_face is not None and face != palifico_restrict_face:
                        face = palifico_restrict_face
                        action = Action.bid(qty, face)
                    current_bid = (qty, face)
                    current_bid_maker = current_player
                else:
                    raise ValueError("Unknown action")
                current_player = self.next_player_idx(current_player, state['dice_counts'])
        alive = [i for i, c in enumerate(state['dice_counts']) if c > 0]
        winner = alive[0] if len(alive) >= 1 else None
        return winner, state
```

---

## agents/random_agent.py

```python
# agents/random_agent.py
import random
from sim.perudo import Action

class RandomAgent:
    def __init__(self, name='random', seed=None):
        self.name = name
        self.rng = random.Random(seed)

    def select_action(self, obs):
        sim = obs.get('_simulator')
        state = {'dice_counts': obs['dice_counts']}
        actions = sim.legal_actions(state, obs['current_bid'], obs.get('palifico_restrict_face'))
        # avoid call if no bid
        if obs['current_bid'] is None:
            actions = [a for a in actions if a[0] != 'call']
        return self.rng.choice(actions)
```

---

## agents/baseline_agent.py

```python
# agents/baseline_agent.py
from collections import Counter
from sim.perudo import Action, binom_cdf_ge

class BaselineAgent:
    def __init__(self, name='baseline', threshold_call=0.5):
        self.name = name
        self.threshold_call = threshold_call

    def select_action(self, obs):
        sim = obs.get('_simulator')
        my_hand = obs['my_hand']
        current_bid = obs['current_bid']
        TD = sum(obs['dice_counts'])
        if current_bid is None:
            face_counts = Counter(my_hand)
            if len(face_counts) == 0:
                return Action.call()
            face = max(face_counts, key=face_counts.get)
            return Action.bid(1, face)
        qty, face = current_bid
        k = sum(1 for d in my_hand if d == face)
        n_other = TD - len(my_hand)
        if face != 1 and (sim.ones_are_wild and not (obs['palifico_active'] and obs['dice_counts'][obs['player_idx']]==1)):
            p = 1/3
        else:
            p = 1/6
        need = max(0, qty - k)
        P_true = binom_cdf_ge(n_other, need, p)
        if P_true < self.threshold_call:
            return Action.call()
        # minimal raise: increase qty by 1
        if qty + 1 <= TD:
            return Action.bid(qty + 1, face)
        for f in range(face + 1, 7):
            return Action.bid(qty, f)
        return Action.call()
```

---

## agents/mc_agent.py

```python
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
```

---

## eval/tournament.py

```python
# eval/tournament.py
import argparse
from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent
from collections import Counter

AGENTS = {
    'random': RandomAgent,
    'baseline': BaselineAgent,
    'mc': MonteCarloAgent,
}


def make_agent(name, sim, mc_n=200):
    cls = AGENTS.get(name)
    if cls is None:
        raise ValueError('Unknown agent')
    return cls()


def play_match(sim, agent_names, games=100, verbose=False, mc_n=200):
    results = Counter()
    for g in range(games):
        # instantiate agents
        agents = []
        for i in range(sim.num_players):
            # alternate players between agent_names
            agent_name = agent_names[i % len(agent_names)]
            a = make_agent(agent_name, sim, mc_n=mc_n)
            agents.append(a)
        winner, _ = sim.play_game(agents, verbose=verbose)
        results[winner] += 1
        if (g + 1) % 10 == 0:
            print(f"Played {g+1}/{games}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--players', type=int, default=3)
    parser.add_argument('--games', type=int, default=200)
    parser.add_argument('--agent1', type=str, default='baseline')
    parser.add_argument('--agent2', type=str, default='mc')
    parser.add_argument('--mc-n', type=int, default=200)
    parser.add_argument('--palifico', action='store_true')
    parser.add_argument('--exact', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    sim = PerudoSimulator(num_players=args.players, start_dice=5, ones_are_wild=True, use_palifico=args.palifico, use_exact=args.exact)
    results = play_match(sim, [args.agent1, args.agent2], games=args.games, verbose=args.verbose, mc_n=args.mc_n)
    print('Results (winner idx counts):', results)
```

---

## requirements.txt

```
# minimal
numpy
tqdm

# optional (if you implement NN improvements)
# torch
```

---

## README.md (short)

```
Perudo project template

1. Save files under the structure shown above.
2. Create a virtualenv and `pip install -r requirements.txt`.
3. Run tournaments:
   python eval/tournament.py --agent1 baseline --agent2 mc --games 200 --mc-n 300 --palifico --exact

Notes:
- The Monte-Carlo agent is single-machine friendly; for speed, you can parallelize evaluate_action in mc_agent using multiprocessing.
- Use this modular layout to later plug ISMCTS, opponent modeling, or NN-based rollout policies.
```

---

### Next steps I can do for you
- Implement multiprocessing in `agents/mc_agent.py` evaluation (safe pickling patterns).
- Add ISMCTS module `agents/ismcts_agent.py` (PUCT, priors, progressive widening) and a ready-to-run implementation.
- Add small PyTorch policy/value network + training scaffold (for rollout policy improvement).

Which of the three next steps do you want me to implement now?
