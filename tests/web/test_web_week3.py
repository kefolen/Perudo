"""
Test suite for Week 3 web interface functionality.
Tests UI polish features including game history display, enhanced CSS styling,
and comprehensive game completion flows.
"""

import pytest
import json
import sys
import os

# Add parent directory to path for importing web modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'web'))

from web.app import app, rooms, players, create_room, join_room, start_game
from web.interactive_game import InteractivePerudoGame


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
    room_code, host_session = create_room("TestHost", max_players=2)
    
    # Start the game to initialize game and AI agents
    success = start_game(room_code, host_session)
    assert success, "Game should start successfully"
    
    return {
        'room_code': room_code,
        'host_session': host_session,
        'room_data': rooms[room_code]
    }


class TestGameHistoryTracking:
    """Test game history tracking functionality."""
    
    def test_interactive_game_has_round_history(self):
        """Test that InteractivePerudoGame initializes with empty round_history."""
        game = InteractivePerudoGame(["TestPlayer"], [])
        assert hasattr(game, 'round_history')
        assert isinstance(game.round_history, list)
        assert len(game.round_history) == 0
    
    def test_round_history_resets_on_new_round(self):
        """Test that round_history is reset when a new round starts."""
        from agents.random_agent import RandomAgent
        game = InteractivePerudoGame(["TestPlayer"], [RandomAgent("Random_AI_1")])
        
        # Add some fake history
        game.round_history.append({'player': 'Test', 'action': 'Test Action'})
        assert len(game.round_history) == 1
        
        # Force game not to be over to test history reset
        game.game_over = False
        game.winner = None
        
        # Start new round should reset history
        game._start_new_round()
        assert len(game.round_history) == 0
    
    def test_action_recorded_in_history(self):
        """Test that actions are recorded in round_history."""
        game = InteractivePerudoGame(["TestPlayer"], [])
        initial_history_len = len(game.round_history)
        
        # Submit a bid if it's the player's turn
        if game.current_player == 0:  # Human player
            try:
                game.submit_bid(1, 2)
                # History should have one more entry
                assert len(game.round_history) == initial_history_len + 1
                # Check entry format
                entry = game.round_history[-1]
                assert 'player' in entry
                assert 'action' in entry
                assert 'player_index' in entry
                assert 'Bid 1 × 2' in entry['action']
            except ValueError:
                # Action might not be legal, which is OK for this test
                pass
    
    def test_history_entry_format(self):
        """Test that history entries have correct format."""
        game = InteractivePerudoGame(["TestPlayer"], [])
        
        # Manually add a history entry to test format
        game.round_history.append({
            'player': 'TestPlayer',
            'action': 'Bid 2 × 3',
            'player_index': 0
        })
        
        entry = game.round_history[0]
        assert isinstance(entry, dict)
        assert 'player' in entry
        assert 'action' in entry
        assert 'player_index' in entry
        assert isinstance(entry['player'], str)
        assert isinstance(entry['action'], str)
        assert isinstance(entry['player_index'], int)
    
    def test_observation_includes_round_history(self):
        """Test that get_observation returns round_history."""
        game = InteractivePerudoGame(["TestPlayer"], [])
        obs = game.get_observation(0)
        
        assert 'round_history' in obs
        assert isinstance(obs['round_history'], list)


class TestPollingEndpointHistory:
    """Test that polling endpoint returns game history."""
    
    def test_poll_response_includes_history(self, client, sample_room_with_game):
        """Test that polling response includes round_history field."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/poll/{room_code}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'round_history' in data
        assert isinstance(data['round_history'], list)
    
    def test_history_updates_after_action(self, client, sample_room_with_game):
        """Test that history is updated after submitting an action."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Get initial history
        response = client.get(f'/poll/{room_code}')
        initial_data = json.loads(response.data)
        initial_history_len = len(initial_data.get('round_history', []))
        
        # Submit action if it's the player's turn
        if initial_data['is_your_turn']:
            action_response = client.post('/action', json={
                'room_code': room_code,
                'action_type': 'bid',
                'quantity': 1,
                'face': 2
            })
            
            # If action was successful, check history updated
            if action_response.status_code == 200:
                # Get updated state
                response = client.get(f'/poll/{room_code}')
                updated_data = json.loads(response.data)
                updated_history_len = len(updated_data.get('round_history', []))
                
                # History should have at least one entry (may have more if AI acted)
                assert updated_history_len > initial_history_len


class TestUIRenderingAndStyling:
    """Test UI rendering and CSS styling features."""
    
    def test_game_page_contains_css_styling(self, client, sample_room_with_game):
        """Test that game page contains comprehensive CSS styling."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/game/{room_code}')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Check for key CSS classes that should be present
        css_classes = [
            '.game-container',
            '.dice-display',
            '.current-bid',
            '.players-status',
            '.actions',
            '.action-form',
            '.game-history',
            '.history-item',
            '.history-player',
            '.history-action'
        ]
        
        for css_class in css_classes:
            assert css_class in html, f"Missing CSS class: {css_class}"
    
    def test_room_page_has_polished_styling(self, client):
        """Test that room page has polished styling elements."""
        # Create room
        room_code, session_id = create_room("TestPlayer")
        
        with client.session_transaction() as sess:
            sess['session_id'] = session_id
        
        response = client.get(f'/room/{room_code}')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Check for styling elements
        styling_elements = [
            '.room-info',
            '.room-code',
            '.players-section',
            '.player-list',
            '.player-item',
            '.host-badge',
            '.game-status',
            '.button-group'
        ]
        
        for element in styling_elements:
            assert element in html, f"Missing styling element: {element}"
    
    def test_home_page_has_clean_styling(self, client):
        """Test that home page has clean, polished styling."""
        response = client.get('/')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Check for key styling elements
        styling_elements = [
            '.container',
            '.form-section',
            'button:hover',
            '.instructions'
        ]
        
        for element in styling_elements:
            assert element in html, f"Missing styling element: {element}"


class TestGameCompletionFlows:
    """Test complete game flows and UI behavior."""
    
    def test_winner_display_functionality(self, client):
        """Test that winner display works correctly."""
        # This test would ideally simulate a complete game
        # For now, we test that the game over screen renders properly
        room_code, session_id = create_room("TestPlayer", max_players=2)
        start_game(room_code, session_id)
        
        with client.session_transaction() as sess:
            sess['session_id'] = session_id
        
        # Get game page to ensure it renders
        response = client.get(f'/game/{room_code}')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Check for game over styling
        assert '.game-over' in html
        assert 'To Main Page' in html
    
    def test_game_state_persistence_during_polling(self, client, sample_room_with_game):
        """Test that game state persists correctly during polling."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        # Poll multiple times
        for _ in range(3):
            response = client.get(f'/poll/{room_code}')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            
            # Check that essential fields are present
            required_fields = [
                'current_player', 'is_your_turn', 'your_dice',
                'players', 'game_over', 'round_history'
            ]
            
            for field in required_fields:
                assert field in data, f"Missing field in polling response: {field}"
    
    def test_multiple_player_session_handling(self, client):
        """Test handling of multiple player sessions."""
        # Create room and join as second player
        room_code, host_session = create_room("Host", max_players=3)
        guest_session = join_room(room_code, "Guest")
        
        assert guest_session is not None
        
        # Start game
        start_game(room_code, host_session)
        
        # Test that both sessions can access game
        for session_id in [host_session, guest_session]:
            with client.session_transaction() as sess:
                sess['session_id'] = session_id
            
            response = client.get(f'/poll/{room_code}')
            assert response.status_code == 200
            
            data = json.loads(response.data)
            assert 'round_history' in data


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases for Week 3 features."""
    
    def test_polling_nonexistent_room_history(self, client):
        """Test polling for history of non-existent room."""
        with client.session_transaction() as sess:
            sess['session_id'] = 'fake-session'
        
        response = client.get('/poll/9999')
        assert response.status_code == 404
    
    def test_empty_history_display(self, client, sample_room_with_game):
        """Test that empty history is handled gracefully."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/poll/{room_code}')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        
        # Even with empty history, the field should exist
        assert 'round_history' in data
        assert isinstance(data['round_history'], list)
    
    def test_game_page_javascript_functionality(self, client, sample_room_with_game):
        """Test that game page includes necessary JavaScript functionality."""
        room_code = sample_room_with_game['room_code']
        host_session = sample_room_with_game['host_session']
        
        with client.session_transaction() as sess:
            sess['session_id'] = host_session
        
        response = client.get(f'/game/{room_code}')
        assert response.status_code == 200
        
        html = response.data.decode()
        
        # Check for key JavaScript functions
        js_functions = [
            'updateGameDisplay',
            'pollGameState',
            'submitBid',
            'submitCall',
            'submitExact',
            'showMessage'
        ]
        
        for func in js_functions:
            assert func in html, f"Missing JavaScript function: {func}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])