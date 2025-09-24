import random
import math
from functools import lru_cache


# small helpers
@lru_cache(maxsize=1024)
def comb(n, k):
    return math.comb(n, k) if 0 <= k <= n else 0


_BINOM_TAILS = {}  # e.g. { 1/6: [ None, [tail for n=1], [tail for n=2], ... ], 1/3: ... }


def _build_table_for_p(p, max_n):
    """Build table for single p up to n=max_n (inclusive)."""
    tables = [None] * (max_n + 1)
    for n in range(max_n + 1):
        # compute pmf then tail (reverse cumulative)
        # pmf length = n+1
        pmf = [0.0] * (n + 1)
        for k in range(n + 1):
            pmf[k] = math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
        tail = [0.0] * (n + 1)
        s = 0.0
        for k in range(n, -1, -1):
            s += pmf[k]
            tail[k] = s
        tables[n] = tail
    return tables


def precompute_binom_tails(max_n=60, ps=(1 / 6, 1 / 3)):
    """
    Precompute binomial tail tables for p values in `ps` up to n = max_n.
    - max_n: maximum number of Bernoulli trials to support (typical Perudo: <= 40)
    - ps: iterable of p values to precompute (defaults to 1/6 and 1/3)

    After calling this, use binom_cdf_ge_fast(n,k,p) for O(1) lookup.
    """
    global _BINOM_TAILS
    for p in ps:
        # normalize float key for robustness (use exact fractional values where possible)
        key = float(p)
        # if already have table but it's smaller, expand
        if key in _BINOM_TAILS and len(_BINOM_TAILS[key]) - 1 >= max_n:
            continue
        _BINOM_TAILS[key] = _build_table_for_p(key, max_n)
    return _BINOM_TAILS

precompute_binom_tails(max_n=50, ps=(1/6, 1/3))


def binom_cdf_ge_fast(n, k, p):
    """
    Fast lookup of P(X >= k) for X ~ Binomial(n, p).
    - If we have precomputed table for this p and n, returns table value.
    - Otherwise falls back to direct sum (slower).
    """
    # canonicalize p key for lookup
    p_key = float(p)
    if p_key in _BINOM_TAILS:
        table = _BINOM_TAILS[p_key]
        if 0 <= n < len(table):
            kk = max(0, min(k, n))
            return table[n][kk]
    # fallback: direct computation (slower). We keep original comb cached.
    if n < 0:
        return 0.0
    s = 0.0
    # ensure k within bounds
    k0 = max(0, min(k, n))
    for j in range(k0, n + 1):
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
    def __init__(self, num_players=3, start_dice=5, ones_are_wild=True, use_maputa=True, use_exact=True, seed=None):
        self.num_players = num_players
        self.start_dice = start_dice
        self.ones_are_wild = ones_are_wild
        self.use_maputa = use_maputa
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

    def legal_actions(self, state, current_bid, maputa_restrict_face=None):
        actions = []
        TD = self.total_dice(state)
        if TD <= 0:
            return [Action.call()]
        for q, f in self.legal_bids_after(current_bid, TD):
            if maputa_restrict_face is not None and f != maputa_restrict_face:
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

    def play_game(self, agents):
        state = self.new_game()
        starting_player = 0
        while sum(1 for c in state['dice_counts'] if c > 0) > 1:
            alive = [i for i, c in enumerate(state['dice_counts']) if c > 0]
            while starting_player not in alive:
                starting_player = self.next_player_idx(starting_player, state['dice_counts'])
            maputa_active = self.use_maputa and (state['dice_counts'][starting_player] == 1)
            maputa_restrict_face = None
            hands = self.roll_hands(state)
            current_bid = None
            current_bid_maker = None
            current_player = starting_player
            first_bid_by = None
            loser = None
            while loser is None:
                obs = {
                    'player_idx': current_player,
                    'my_hand': list(hands[current_player]),
                    'dice_counts': list(state['dice_counts']),
                    'current_bid': current_bid,
                    'history': [],
                    'maputa_active': maputa_active,
                    'maputa_restrict_face': maputa_restrict_face,
                    '_simulator': self,
                }
                action = agents[current_player].select_action(obs)
                if Action.is_bid(action) and first_bid_by is None:
                    first_bid_by = current_player
                    if maputa_active:
                        maputa_restrict_face = Action.face(action)

                if action[0] == 'bid':
                    qty, face = action[1], action[2]
                    if maputa_restrict_face is not None and face != maputa_restrict_face:
                        # invalid
                        loser = current_player
                    else:
                        current_bid = (qty, face)
                        current_bid_maker = current_player
                        current_player = self.next_player_idx(current_player, state['dice_counts'])
                else:
                    if current_bid is None:
                        # invalid
                        loser = current_player
                    else:
                        bid_true, cnt = self.is_bid_true(hands, current_bid,
                                                         ones_are_wild=(self.ones_are_wild and not maputa_active))

                        if action[0] == 'call':
                            if bid_true: loser = current_player
                            else: loser = current_bid_maker

                        else: #action[0] == exact
                            if cnt == current_bid[0]:
                                loser = current_bid_maker
                                state['dice_counts'][current_player] = state['dice_counts'][current_player] + 1
                            else:
                                loser = current_player

            state['dice_counts'][loser] = max(0, state['dice_counts'][loser] - 1)
            starting_player = loser

        alive = [i for i, c in enumerate(state['dice_counts']) if c > 0]
        winner = alive[0] if len(alive) >= 1 else None
        return winner, state
