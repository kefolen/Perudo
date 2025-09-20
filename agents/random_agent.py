import random
from sim.perudo import Action

class RandomAgent:
    def __init__(self, name='random', seed=None):
        self.name = name
        self.rng = random.Random(seed)

    def select_action(self, obs):
        sim = obs.get('_simulator')
        state = {'dice_counts': obs['dice_counts']}
        actions = sim.legal_actions(state, obs['current_bid'], obs.get('maputa_restrict_face'))
        # avoid call or exact if no bid
        if obs['current_bid'] is None:
            actions = [a for a in actions if a[0] != 'call' and a[0] != 'exact']
        return self.rng.choice(actions)
