"""
Test suite for AI player management system.
Tests room creation with AI pre-population, human replacement of AI players,
agent type configuration, and game initialization with mixed human/AI players.
"""

import pytest
import json
import sys
import os

# Add parent directory to path for importing web modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))

from web.app import (
    app, rooms, players, create_room, join_room, start_game,
    load_mc_config, create_agent_from_config
)
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        with app.app_context():
            # Clear in-memory storage before each test
            rooms.clear()
            players.clear()
            yield client


class TestAIPrePopulation:
    """Test AI player pre-population during room creation."""
    
    def test_create_room_populates_with_ai_players(self, client):
        """Test that creating a room pre-populates with AI players."""
        room_code, session_id = create_room("TestPlayer", max_players=4)
        
        room = rooms[room_code]
        
        # Should have 4 players: 1 human + 3 AI
        assert len(room['players']) == 4
        
        # First player should be human host
        assert room['players'][0]['session_id'] == session_id
        assert room['players'][0]['is_ai'] == False
        
        # Remaining players should be AI
        for i in range(1, 4):
            assert room['players'][i]['is_ai'] == True
            assert 'ai_config' in room['players'][i]
            assert room['players'][i]['ai_config']['type'] == 'random'
            assert room['players'][i]['ai_config']['name'] == f'Random_AI_{i}'
    
    def test_create_room_with_different_player_counts(self, client):
        """Test room creation with different player counts."""
        for max_players in [2, 3, 5, 6, 7, 8]:
            room_code, session_id = create_room("TestPlayer", max_players=max_players)
            
            room = rooms[room_code]
            
            # Should have max_players total
            assert len(room['players']) == max_players
            
            # First should be human, rest AI
            assert room['players'][0]['is_ai'] == False
            for i in range(1, max_players):
                assert room['players'][i]['is_ai'] == True
            
            # Clean up for next iteration
            rooms.clear()
            players.clear()
    
    def test_ai_players_stored_in_players_dict(self, client):
        """Test that AI players are properly stored in players dictionary."""
        room_code, session_id = create_room("TestPlayer", max_players=3)
        
        room = rooms[room_code]
        
        # Check that AI players are in players dict
        for player_entry in room['players'][1:]:  # Skip human host
            ai_session_id = player_entry['session_id']
            assert ai_session_id in players
            
            ai_player_data = players[ai_session_id]
            assert ai_player_data['is_ai'] == True
            assert 'ai_config' in ai_player_data
            assert ai_player_data['room_code'] == room_code
    
    def test_custom_ai_configs(self, client):
        """Test creating room with custom AI configurations."""
        custom_ai_configs = [
            {'type': 'baseline', 'name': 'Smart_AI_1', 'is_ai': True},
            {'type': 'mc', 'name': 'Advanced_AI_2', 'is_ai': True}
        ]
        
        room_code, session_id = create_room("TestPlayer", max_players=3, ai_configs=custom_ai_configs)
        
        room = rooms[room_code]
        
        # Check AI configurations
        assert room['players'][1]['ai_config']['type'] == 'baseline'
        assert room['players'][1]['ai_config']['name'] == 'Smart_AI_1'
        assert room['players'][2]['ai_config']['type'] == 'mc'
        assert room['players'][2]['ai_config']['name'] == 'Advanced_AI_2'


class TestHumanAIReplacement:
    """Test human players replacing AI players when joining."""
    
    def test_join_room_replaces_topmost_ai(self, client):
        """Test that joining a room replaces the topmost AI player."""
        # Create room with 3 players (1 human, 2 AI)
        room_code, host_session = create_room("Host", max_players=3)
        
        # Join as second player
        guest_session = join_room(room_code, "Guest")
        assert guest_session is not None
        
        room = rooms[room_code]
        
        # Should still have 3 players total
        assert len(room['players']) == 3
        
        # First player should still be host
        assert room['players'][0]['session_id'] == host_session
        assert room['players'][0]['is_ai'] == False
        
        # Second player should now be the guest (replaced first AI)
        assert room['players'][1]['session_id'] == guest_session
        assert room['players'][1]['is_ai'] == False
        
        # Third player should still be AI
        assert room['players'][2]['is_ai'] == True
    
    def test_join_room_updates_players_dict(self, client):
        """Test that joining removes old AI and adds new human in players dict."""
        room_code, host_session = create_room("Host", max_players=3)
        
        # Get original AI session ID
        original_ai_session = rooms[room_code]['players'][1]['session_id']
        assert original_ai_session in players
        
        # Join as guest
        guest_session = join_room(room_code, "Guest")
        
        # Old AI should be removed from players dict
        assert original_ai_session not in players
        
        # New human should be in players dict
        assert guest_session in players
        assert players[guest_session]['nickname'] == "Guest"
        assert players[guest_session]['is_ai'] == False
    
    def test_join_room_no_ai_slots_available(self, client):
        """Test joining when all slots are filled with humans."""
        # Create 2-player room
        room_code, host_session = create_room("Host", max_players=2)
        
        # Fill the AI slot
        guest_session = join_room(room_code, "Guest")
        assert guest_session is not None
        
        # Try to join when no AI slots available
        third_session = join_room(room_code, "Third")
        assert third_session is None
    
    def test_multiple_humans_join_sequentially(self, client):
        """Test multiple humans joining and replacing AI players in order."""
        room_code, host_session = create_room("Host", max_players=4)
        
        # Initially: Host (human), AI_1, AI_2, AI_3
        room = rooms[room_code]
        assert room['players'][0]['is_ai'] == False
        assert room['players'][1]['is_ai'] == True
        assert room['players'][2]['is_ai'] == True
        assert room['players'][3]['is_ai'] == True
        
        # First guest joins, replaces AI_1
        guest1_session = join_room(room_code, "Guest1")
        assert guest1_session is not None
        assert room['players'][1]['session_id'] == guest1_session
        assert room['players'][1]['is_ai'] == False
        
        # Second guest joins, replaces AI_2
        guest2_session = join_room(room_code, "Guest2")
        assert guest2_session is not None
        assert room['players'][2]['session_id'] == guest2_session
        assert room['players'][2]['is_ai'] == False
        
        # Third guest joins, replaces AI_3
        guest3_session = join_room(room_code, "Guest3")
        assert guest3_session is not None
        assert room['players'][3]['session_id'] == guest3_session
        assert room['players'][3]['is_ai'] == False
        
        # No more AI slots available
        guest4_session = join_room(room_code, "Guest4")
        assert guest4_session is None


class TestAgentConfiguration:
    """Test agent type configuration and instantiation."""
    
    def test_load_mc_config(self, client):
        """Test loading MC configuration from JSON file."""
        config = load_mc_config()
        
        # Should have all required parameters
        required_params = [
            'name', 'n', 'chunk_size', 'max_rounds', 'simulate_to_round_end',
            'early_stop_margin', 'weighted_sampling', 'enable_parallel',
            'num_workers', 'enhanced_pruning', 'variance_reduction',
            'betting_history_enabled', 'player_trust_enabled',
            'trust_learning_rate', 'history_memory_rounds', 'bayesian_sampling'
        ]
        
        for param in required_params:
            assert param in config
    
    def test_create_random_agent_from_config(self, client):
        """Test creating RandomAgent from configuration."""
        config = {'type': 'random', 'name': 'Test_Random'}
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, RandomAgent)
        assert agent.name == 'Test_Random'
    
    def test_create_baseline_agent_from_config(self, client):
        """Test creating BaselineAgent from configuration."""
        config = {'type': 'baseline', 'name': 'Test_Baseline'}
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, BaselineAgent)
        assert agent.name == 'Test_Baseline'
    
    def test_create_mc_agent_from_config(self, client):
        """Test creating MonteCarloAgent from configuration."""
        config = {'type': 'mc', 'name': 'Test_MC'}
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, MonteCarloAgent)
        assert agent.name == 'Test_MC'
        # Verify some MC-specific parameters were loaded
        assert hasattr(agent, 'N')
        assert hasattr(agent, 'chunk_size')
        assert hasattr(agent, 'max_rounds')
    
    def test_create_agent_unknown_type_defaults_to_random(self, client):
        """Test that unknown agent types default to RandomAgent."""
        config = {'type': 'unknown', 'name': 'Test_Unknown'}
        agent = create_agent_from_config(config)
        
        assert isinstance(agent, RandomAgent)
        assert agent.name == 'Test_Unknown'


class TestGameInitialization:
    """Test game initialization with mixed human/AI players."""
    
    def test_start_game_with_mixed_players(self, client):
        """Test starting game with human and AI players."""
        # Create room and add human player
        room_code, host_session = create_room("Host", max_players=3)
        guest_session = join_room(room_code, "Guest")
        
        # Start game
        success = start_game(room_code, host_session)
        assert success
        
        room = rooms[room_code]
        game = room['game_state']
        
        assert game is not None
        assert hasattr(game, 'current_player')
        assert game.get_player_count() == 3
        
        # Should have AI agents stored
        assert 'ai_agents' in room
        assert len(room['ai_agents']) == 1  # One AI remaining
    
    def test_start_game_creates_correct_agent_types(self, client):
        """Test that start_game creates agents of configured types."""
        # Create room with mixed AI types
        ai_configs = [
            {'type': 'baseline', 'name': 'Smart_AI', 'is_ai': True},
            {'type': 'mc', 'name': 'Advanced_AI', 'is_ai': True}
        ]
        room_code, host_session = create_room("Host", max_players=3, ai_configs=ai_configs)
        
        # Start game
        start_game(room_code, host_session)
        
        room = rooms[room_code]
        ai_agents = room['ai_agents']
        
        # Should have 2 AI agents of correct types
        assert len(ai_agents) == 2
        assert isinstance(ai_agents[0], BaselineAgent)
        assert ai_agents[0].name == 'Smart_AI'
        assert isinstance(ai_agents[1], MonteCarloAgent)
        assert ai_agents[1].name == 'Advanced_AI'
    
    def test_start_game_preserves_player_order(self, client):
        """Test that player order is preserved during game initialization."""
        room_code, host_session = create_room("Host", max_players=3)
        guest_session = join_room(room_code, "Guest")
        
        # Before game start, verify player order
        room = rooms[room_code]
        assert room['players'][0]['session_id'] == host_session  # Host first
        assert room['players'][1]['session_id'] == guest_session  # Guest second
        assert room['players'][2]['is_ai'] == True  # AI third
        
        # Start game
        start_game(room_code, host_session)
        
        # Game should maintain this order
        game = room['game_state']
        assert game.get_player_name(0) == "Host"
        assert game.get_player_name(1) == "Guest"
        assert game.get_player_name(2).startswith("Random_AI_")


class TestFormIntegration:
    """Test integration with web form submission."""
    
    def test_create_room_endpoint_with_ai_config(self, client):
        """Test room creation endpoint handles AI configuration form data."""
        response = client.post('/create_room', data={
            'nickname': 'TestHost',
            'player_count': '3',
            'ai_1_type': 'baseline',
            'ai_2_type': 'mc'
        })
        
        # Should redirect to room page
        assert response.status_code == 302
        
        # Verify room was created with correct AI configs
        assert len(rooms) == 1
        room_code = list(rooms.keys())[0]
        room = rooms[room_code]
        
        # Should have 3 players with correct types
        assert len(room['players']) == 3
        assert room['players'][0]['is_ai'] == False  # Host
        assert room['players'][1]['ai_config']['type'] == 'baseline'
        assert room['players'][2]['ai_config']['type'] == 'mc'
    
    def test_room_page_displays_ai_types(self, client):
        """Test that room page displays AI player types correctly."""
        # Create room with mixed AI types
        ai_configs = [
            {'type': 'baseline', 'name': 'Smart_AI', 'is_ai': True},
            {'type': 'random', 'name': 'Simple_AI', 'is_ai': True}
        ]
        room_code, session_id = create_room("Host", max_players=3, ai_configs=ai_configs)
        
        with client.session_transaction() as sess:
            sess['session_id'] = session_id
        
        response = client.get(f'/room/{room_code}')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Should display AI types
        assert 'BASELINE AI' in html
        assert 'RANDOM AI' in html


if __name__ == '__main__':
    pytest.main([__file__, '-v'])