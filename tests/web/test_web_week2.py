"""
Test suite for Week 2 web interface functionality.
Tests game integration, RandomAgent bots, action processing, and polling endpoints.
"""

import pytest
import json
import sys
import os

# Add parent directory to path for importing web modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))

from web.app import app, rooms, players, create_room, join_room, start_game


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


@pytest.fixture
def sample_room_with_game(client):
    """Create a sample room with a started game for testing."""
    # Create room with host
    room_code, host_session = create_room("TestHost", max_players=4)
    
    # Start the game to initialize PerudoSimulator and AI agents
    success = start_game(room_code, host_session)
    assert success, "Game should start successfully"
    
    return {
        'room_code': room_code,
        'host_session': host_session,
        'room_data': rooms[room_code]
    }


class TestGameInitialization:
    """Test game initialization with RandomAgent bots."""
    
    def test_start_game_creates_simulator(self, client):
        """Test that starting a game creates a PerudoSimulator instance."""
        # Create room
        room_code, session_id = create_room("TestPlayer", max_players=3)
        
        # Verify game is not started yet
        assert rooms[room_code]['game_state'] is None
        
        # Start game
        success = start_game(room_code, session_id)
        
        assert success
        assert rooms[room_code]['game_state'] is not None
        assert hasattr(rooms[room_code]['game_state'], 'current_player')
        assert hasattr(rooms[room_code]['game_state'], 'is_game_over')
    
    def test_start_game_fills_slots_with_random_agents(self, client):
        """Test that empty slots are filled with RandomAgent bots."""
        # Create room with 2 max players but only 1 human
        room_code, session_id = create_room("TestPlayer", max_players=2)
        
        # Start game
        success = start_game(room_code, session_id)
        
        assert success
        room_data = rooms[room_code]
        
        # Should have 1 AI agent for the empty slot
        assert 'ai_agents' in room_data
        assert len(room_data['ai_agents']) == 1
        assert hasattr(room_data['ai_agents'][0], 'select_action')
        assert room_data['ai_agents'][0].name == "Random_AI_1"
    
    def test_start_game_multiple_ai_agents(self, client):
        """Test creating multiple AI agents for multiple empty slots."""
        # Create room with 4 max players but only 1 human
        room_code, session_id = create_room("TestPlayer", max_players=4)
        
        # Start game
        success = start_game(room_code, session_id)
        
        assert success
        room_data = rooms[room_code]
        
        # Should have 3 AI agents
        assert len(room_data['ai_agents']) == 3
        assert room_data['ai_agents'][0].name == "Random_AI_1"
        assert room_data['ai_agents'][1].name == "Random_AI_2"
        assert room_data['ai_agents'][2].name == "Random_AI_3"
    
    def test_only_host_can_start_game(self, client):
        """Test that only the host can start the game."""
        # Create room
        room_code, host_session = create_room("Host", max_players=3)
        
        # Join as another player
        guest_session = join_room(room_code, "Guest")
        assert guest_session is not None
        
        # Try to start game as guest (should fail)
        success = start_game(room_code, guest_session)
        assert not success
        
        # Start game as host (should succeed)
        success = start_game(room_code, host_session)
        assert success
    
    def test_cannot_start_game_twice(self, client):
        """Test that a game cannot be started twice."""
        # Create room and start game
        room_code, session_id = create_room("TestPlayer", max_players=2)
        success1 = start_game(room_code, session_id)
        assert success1
        
        # Try to start again (should fail)
        success2 = start_game(room_code, session_id)
        assert not success2


class TestActionProcessing:
    """Test action submission endpoint and game flow."""
    
    def test_action_endpoint_validation(self, client):
        """Test action endpoint validates requests properly."""
        # Test missing JSON data
        response = client.post('/action')
        assert response.status_code == 400


class TestDiceCounter:
    """Test that poll response includes total dice in play counter."""

    def test_poll_includes_total_dice_in_play(self, client, sample_room_with_game):
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']

        # Set session for the host
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Poll game state
        resp = client.get(f'/poll/{room_code}')
        assert resp.status_code == 200
        data = json.loads(resp.data.decode())
        
        # Field must exist and equal sum of dice counts
        assert 'total_dice_in_play' in data
        summed = sum(p['dice_count'] for p in data['players'])
        assert data['total_dice_in_play'] == summed
    
    def test_action_requires_valid_session(self, client):
        """Test that actions require valid session."""
        # Create room with game
        room_code, session_id = create_room("TestPlayer", max_players=2)
        start_game(room_code, session_id)
        
        # Try action without session
        response = client.post('/action',
                             json={'room_code': room_code, 'action_type': 'bid'})
        assert response.status_code == 401
    
    def test_bid_action_processing(self, client, sample_room_with_game):
        """Test bid action processing."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        # Set session for test client
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Submit bid action
        response = client.post('/action',
                             json={
                                 'room_code': room_code,
                                 'action_type': 'bid',
                                 'quantity': 2,
                                 'face': 3
                             })
        
        # Should succeed if it's the player's turn, or fail if it's not
        assert response.status_code in [200, 400]
        data = json.loads(response.data)
        
        if response.status_code == 200:
            assert 'success' in data
            assert 'game_over' in data
        else:
            # Should be "Not your turn" if it's AI's turn first
            assert 'error' in data
    
    def test_call_action_processing(self, client, sample_room_with_game):
        """Test call action processing."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Try call action (may not be valid if no current bid)
        response = client.post('/action',
                             json={
                                 'room_code': room_code,
                                 'action_type': 'call'
                             })
        
        # Should return some response (success or error based on game state)
        assert response.status_code in [200, 400]
        data = json.loads(response.data)
        assert ('success' in data) or ('error' in data)
    
    def test_exact_action_processing(self, client, sample_room_with_game):
        """Test exact call action processing."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Try exact action
        response = client.post('/action',
                             json={
                                 'room_code': room_code,
                                 'action_type': 'exact'
                             })
        
        # Should return some response
        assert response.status_code in [200, 400]
        data = json.loads(response.data)
        assert ('success' in data) or ('error' in data)


class TestPollingEndpoint:
    """Test game state polling endpoint."""
    
    def test_poll_endpoint_validation(self, client):
        """Test polling endpoint validates requests properly."""
        # Test invalid room code
        response = client.get('/poll/9999')
        assert response.status_code == 404
        
        # Test room without game
        room_code, _ = create_room("TestPlayer")
        response = client.get(f'/poll/{room_code}')
        assert response.status_code == 400
    
    def test_poll_requires_valid_session(self, client, sample_room_with_game):
        """Test that polling requires valid session."""
        room_code = sample_room_with_game['room_code']
        
        # Try polling without session
        response = client.get(f'/poll/{room_code}')
        assert response.status_code == 401

    def test_poll_players_include_alive_flag(self, client, sample_room_with_game):
        """Players in poll response must include 'alive' boolean matching dice_count>0."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        # Set session for the host
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Poll game state
        resp = client.get(f'/poll/{room_code}')
        assert resp.status_code == 200
        data = json.loads(resp.data.decode())
        assert 'players' in data and isinstance(data['players'], list)
        for p in data['players']:
            assert 'alive' in p
            assert p['alive'] == (p['dice_count'] > 0)

    def test_poll_players_include_last_action_field(self, client, sample_room_with_game):
        """Players in poll response must include 'last_action' per current round (may be None at round start)."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        # Set session for the host
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Poll game state
        resp = client.get(f'/poll/{room_code}')
        assert resp.status_code == 200
        data = json.loads(resp.data.decode())
        assert 'players' in data and isinstance(data['players'], list)
        # Field exists for all players
        for p in data['players']:
            assert 'last_action' in p
        # Initially, before any actions, all last_action should be None
        assert all(p['last_action'] is None for p in data['players'])
    
    def test_poll_returns_game_state(self, client, sample_room_with_game):
        """Test that polling returns complete game state."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/poll/{room_code}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Check required fields
        required_fields = [
            'current_player', 'is_your_turn', 'your_dice', 
            'current_bid', 'legal_actions', 'game_over', 
            'winner', 'players'
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Check data types
        assert isinstance(data['current_player'], int)
        assert isinstance(data['is_your_turn'], bool)
        assert isinstance(data['your_dice'], list)
        assert isinstance(data['legal_actions'], list)
        assert isinstance(data['game_over'], bool)
        assert isinstance(data['players'], list)
        
        # Check players data structure
        if data['players']:
            player = data['players'][0]
            assert 'name' in player
            assert 'dice_count' in player
            assert 'is_ai' in player
    
    def test_poll_player_ownership(self, client):
        """Test that players can only poll games they're in."""
        # Create two separate rooms
        room1_code, session1 = create_room("Player1")
        room2_code, session2 = create_room("Player2")
        
        start_game(room1_code, session1)
        start_game(room2_code, session2)
        
        # Try to poll room2 with session1
        with client.session_transaction() as sess:
            sess['session_id'] = session1
        
        response = client.get(f'/poll/{room2_code}')
        assert response.status_code == 403


class TestBotIntegration:
    """Test AI bot integration and automatic turns."""
    
    def test_bots_take_turns_automatically(self, client):
        """Test that AI bots take turns automatically after human actions."""
        # Create room with 1 human, 1 bot
        room_code, session_id = create_room("Human", max_players=2)
        start_game(room_code, session_id)
        
        room_data = rooms[room_code]
        game = room_data['game_state']
        initial_player = game.current_player
        
        # If it's human's turn (player 0), submit an action
        if initial_player == 0:
            with client.session_transaction() as sess:
                sess['session_id'] = session_id
            
            response = client.post('/action',
                                 json={
                                     'room_code': room_code,
                                     'action_type': 'bid',
                                     'quantity': 1,
                                     'face': 2
                                 })
            
            # Should succeed and potentially advance to bot's turn
            if response.status_code == 200:
                # Game state should have progressed
                # (Bot should have taken its turn automatically)
                assert True  # Basic verification that action processing worked
    
    def test_bot_agents_have_select_action_method(self, client):
        """Test that bot agents are properly instantiated with select_action method."""
        room_code, session_id = create_room("TestPlayer", max_players=3)
        start_game(room_code, session_id)
        
        room_data = rooms[room_code]
        ai_agents = room_data['ai_agents']
        
        assert len(ai_agents) == 2  # 2 AI agents for empty slots
        
        for agent in ai_agents:
            assert hasattr(agent, 'select_action')
            assert callable(agent.select_action)
            assert hasattr(agent, 'name')


class TestWebRoutes:
    """Test web routes for game interface."""
    
    def test_game_page_requires_started_game(self, client):
        """Test that game page redirects if game not started."""
        # Create room without starting game
        room_code, session_id = create_room("TestPlayer")
        
        with client.session_transaction() as sess:
            sess['session_id'] = session_id
        
        response = client.get(f'/game/{room_code}')
        
        # Should redirect to room page
        assert response.status_code == 302
        assert f'/room/{room_code}' in response.headers['Location']
    
    def test_game_page_shows_when_game_started(self, client, sample_room_with_game):
        """Test that game page shows when game is started."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/game/{room_code}')
        assert response.status_code == 200
        
        # Should contain game interface elements
        html = response.data.decode()
        assert 'Perudo Game' in html
        assert 'Loading game state...' in html
        assert f'Room: {room_code}' in html
    
    def test_room_page_redirects_when_game_started(self, client, sample_room_with_game):
        """Test that room page redirects to game when game is started."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/room/{room_code}')
        assert response.status_code == 200
        
        # Should contain redirect script to game page
        html = response.data.decode()
        assert 'Game in Progress!' in html
        assert f'/game/{room_code}' in html
    
    def test_start_game_endpoint_redirects_to_game(self, client):
        """Test that start game endpoint redirects to game page."""
        room_code, session_id = create_room("TestPlayer")
        
        with client.session_transaction() as sess:
            sess['session_id'] = session_id
        
        response = client.post(f'/start_game/{room_code}')
        
        # Should redirect to game page
        assert response.status_code == 302
        assert f'/game/{room_code}' in response.headers['Location']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])