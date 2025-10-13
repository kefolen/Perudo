import pytest
import random
from sim.perudo import PerudoSimulator, Action
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent


@pytest.fixture
def deterministic_seed():
    """Provides fixed random seed for reproducible tests"""
    return 42


@pytest.fixture
def sample_simulator():
    """Provides a standard PerudoSimulator instance for testing"""
    return PerudoSimulator(num_players=3, start_dice=5, seed=42)


@pytest.fixture
def sample_game_state():
    """Provides a standard game state for testing"""
    return {
        'dice_counts': [5, 5, 5],  # 3 players with 5 dice each
        'player_idx': 0,
        'my_hand': [1, 2, 3, 4, 5],  # Sample hand
        'current_bid': None,
        'maputa_restrict_face': None
    }


@pytest.fixture
def mid_game_state():
    """Provides a mid-game state with fewer dice"""
    return {
        'dice_counts': [3, 2, 3],  # Mid-game with fewer dice
        'player_idx': 1,
        'my_hand': [1, 4],  # Smaller hand
        'current_bid': (2, 3),  # Existing bid
        'maputa_restrict_face': None
    }


@pytest.fixture
def end_game_state():
    """Provides an end-game state with single dice"""
    return {
        'dice_counts': [1, 1, 2],  # End-game scenario
        'player_idx': 0,
        'my_hand': [6],  # Single die
        'current_bid': (1, 2),
        'maputa_restrict_face': 2  # Maputa restriction
    }


@pytest.fixture
def random_agent(deterministic_seed):
    """Provides a RandomAgent instance with fixed seed"""
    return RandomAgent(name='test_random', seed=deterministic_seed)


@pytest.fixture
def baseline_agent():
    """Provides a BaselineAgent instance"""
    return BaselineAgent(name='test_baseline', threshold_call=0.5)


@pytest.fixture
def mc_agent(deterministic_seed):
    """Provides a MonteCarloAgent instance with reduced parameters for testing"""
    import random
    rng = random.Random(deterministic_seed)
    return MonteCarloAgent(
        name='test_mc',
        n=10,  # Reduced for faster testing
        chunk_size=5,
        rng=rng
    )


@pytest.fixture
def all_agent_types(random_agent, baseline_agent, mc_agent):
    """Provides instances of all agent types"""
    return {
        'random': random_agent,
        'baseline': baseline_agent,
        'mc': mc_agent
    }


@pytest.fixture
def sample_hands():
    """Provides various sample hands for testing"""
    return {
        'all_ones': [1, 1, 1, 1, 1],
        'mixed': [1, 2, 3, 4, 5],
        'pairs': [2, 2, 3, 3, 4],
        'single_die': [6],
        'no_ones': [2, 3, 4, 5, 6]
    }


@pytest.fixture
def sample_bids():
    """Provides various sample bids for testing"""
    return {
        'low_bid': (1, 2),
        'medium_bid': (3, 4),
        'high_bid': (8, 6),
        'ones_bid': (2, 1),
        'max_bid': (15, 6)  # For 3 players with 5 dice each
    }


@pytest.fixture
def sample_actions():
    """Provides various Action instances for testing"""
    return {
        'bid_low': Action.bid(1, 2),
        'bid_medium': Action.bid(3, 4),
        'bid_high': Action.bid(8, 6),
        'call': Action.call(),
        'exact': Action.exact()
    }


@pytest.fixture
def observation_with_simulator(sample_simulator, sample_game_state):
    """Provides a complete observation with simulator reference"""
    obs = sample_game_state.copy()
    obs['_simulator'] = sample_simulator
    return obs
