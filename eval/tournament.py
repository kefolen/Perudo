from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent
from collections import Counter
import time

# Configuration constants (modify these values as needed)
NUM_PLAYERS = 6
NUM_GAMES = 30
MC_N = 1000
MAX_ROUNDS = 10
AGENT_NAMES = ['mc', 'baseline', 'baseline', 'baseline', 'baseline', 'baseline']
USE_MAPUTA = True
USE_EXACT = True

AGENTS = {
    'random': RandomAgent,
    'baseline': BaselineAgent,
    'mc': MonteCarloAgent,
}


def make_agent(name, sim, mc_n=200, max_rounds=6):
    cls = AGENTS.get(name)
    if cls is None:
        raise ValueError('Unknown agent')
    if name == 'mc':
        return cls(n=mc_n, max_rounds=max_rounds)
    return cls()


def play_match(sim, agent_names, games=100, mc_n=200, max_rounds=6):
    results = Counter()
    start_time = time.time()
    game_times = []

    for g in range(games):
        # instantiate agents
        agents = []
        for i in range(sim.num_players):
            # alternate players between agent_names
            agent_name = agent_names[i % len(agent_names)]
            a = make_agent(agent_name, sim, mc_n=mc_n, max_rounds=max_rounds)
            agents.append(a)

        game_start_time = time.time()
        winner, _ = sim.play_game(agents)
        game_end_time = time.time()
        game_time = game_end_time - game_start_time
        game_times.append(game_time)

        results[winner] += 1
        if (g + 1) % 10 == 0:
            print(f"Played {g + 1}/{games}, last game time: {game_time:.2f}s")

    total_time = time.time() - start_time
    avg_time = sum(game_times) / len(game_times) if game_times else 0

    print(f"\nTime statistics:")
    print(f"Total simulation time: {total_time:.2f} seconds")
    print(f"Average time per game: {avg_time:.2f} seconds")
    print(f"MC_N parameter: {mc_n}")

    return results


if __name__ == '__main__':
    sim = PerudoSimulator(num_players=NUM_PLAYERS, start_dice=5, ones_are_wild=True, use_maputa=USE_MAPUTA,
                          use_exact=USE_EXACT)
    results = play_match(sim, AGENT_NAMES, games=NUM_GAMES, mc_n=MC_N, max_rounds=MAX_ROUNDS)
    print('Results (winner idx counts):', results)
