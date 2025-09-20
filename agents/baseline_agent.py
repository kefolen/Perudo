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
        if face != 1 and (sim.ones_are_wild and not (obs['maputa_active'] and obs['dice_counts'][obs['player_idx']]==1)):
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
