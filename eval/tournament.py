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


def play_match(sim, agent_names, games=100, mc_n=200):
    results = Counter()
    for g in range(games):
        # instantiate agents
        agents = []
        for i in range(sim.num_players):
            # alternate players between agent_names
            agent_name = agent_names[i % len(agent_names)]
            a = make_agent(agent_name, sim, mc_n=mc_n)
            agents.append(a)
        winner, _ = sim.play_game(agents)
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
    parser.add_argument('--maputa', action='store_true')
    parser.add_argument('--exact', action='store_true')
    args = parser.parse_args()

    sim = PerudoSimulator(num_players=args.players, start_dice=5, ones_are_wild=True, use_maputa=args.maputa, use_exact=args.exact)
    results = play_match(sim, [args.agent1, args.agent2], games=args.games, mc_n=args.mc_n)
    print('Results (winner idx counts):', results)
