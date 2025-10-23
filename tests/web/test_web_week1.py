"""
Tests for Week 1 web interface functionality - Room creation and joining.
Following TDD principles: These tests should fail initially, then pass after implementation.
"""

import unittest
import sys
import os

# Add the project root to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from unittest.mock import patch, MagicMock
import uuid


class TestRoomManagement(unittest.TestCase):
    """Test room creation and joining logic"""
    
    def setUp(self):
        """Set up test fixtures"""
        # These will be imported from web.app once created
        self.rooms = {}
        self.players = {}
    
    def test_create_room_generates_unique_code(self):
        """Test that create_room generates a unique 4-digit room code"""
        # This test will fail until we implement create_room function
        from web.app import create_room
        
        nickname = "TestPlayer"
        room_code, session_id = create_room(nickname)
        
        # Room code should be 4-digit integer
        self.assertIsInstance(room_code, int)
        self.assertGreaterEqual(room_code, 1000)
        self.assertLessEqual(room_code, 9999)
        
        # Session ID should be 8-character string
        self.assertIsInstance(session_id, str)
        self.assertEqual(len(session_id), 8)
    
    def test_create_room_stores_data_correctly(self):
        """Test that create_room stores room and player data correctly"""
        from web.app import create_room, rooms, players
        
        nickname = "TestPlayer"
        room_code, session_id = create_room(nickname)
        
        # Room should be created with correct structure
        self.assertIn(room_code, rooms)
        room = rooms[room_code]
        self.assertEqual(room['players'], [session_id])
        self.assertIsNone(room['game_state'])
        self.assertEqual(room['host'], session_id)
        
        # Player should be stored correctly
        self.assertIn(session_id, players)
        player = players[session_id]
        self.assertEqual(player['nickname'], nickname)
        self.assertEqual(player['room_code'], room_code)
    
    def test_join_room_with_valid_code(self):
        """Test joining an existing room with valid code"""
        from web.app import create_room, join_room, rooms, players
        
        # Create a room first
        host_nickname = "Host"
        room_code, host_session = create_room(host_nickname)
        
        # Join the room
        joiner_nickname = "Joiner"
        joiner_session = join_room(room_code, joiner_nickname)
        
        # Session ID should be 8-character string
        self.assertIsInstance(joiner_session, str)
        self.assertEqual(len(joiner_session), 8)
        
        # Room should now have 2 players
        self.assertEqual(len(rooms[room_code]['players']), 2)
        self.assertIn(joiner_session, rooms[room_code]['players'])
        
        # Joiner should be stored correctly
        self.assertIn(joiner_session, players)
        joiner = players[joiner_session]
        self.assertEqual(joiner['nickname'], joiner_nickname)
        self.assertEqual(joiner['room_code'], room_code)
    
    def test_join_room_with_invalid_code(self):
        """Test joining a non-existent room returns None"""
        from web.app import join_room
        
        invalid_code = 9999
        result = join_room(invalid_code, "TestPlayer")
        self.assertIsNone(result)
    
    def test_room_code_uniqueness(self):
        """Test that multiple room creations generate different codes"""
        from web.app import create_room
        
        codes = set()
        for i in range(10):
            room_code, _ = create_room(f"Player{i}")
            codes.add(room_code)
        
        # All codes should be unique
        self.assertEqual(len(codes), 10)


class TestFlaskRoutes(unittest.TestCase):
    """Test Flask application routes and endpoints"""
    
    def setUp(self):
        """Set up Flask test client"""
        # This will fail until we create the Flask app
        from web.app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_home_page_loads(self):
        """Test that home page (/) loads successfully"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Perudo', response.data)
    
    def test_create_room_endpoint(self):
        """Test POST /create_room endpoint"""
        response = self.app.post('/create_room', data={'nickname': 'TestPlayer'})
        
        # Should redirect to room page
        self.assertEqual(response.status_code, 302)
        
        # Location header should contain room code
        location = response.headers.get('Location')
        self.assertIsNotNone(location)
        self.assertIn('/room/', location)
    
    def test_join_room_endpoint_valid(self):
        """Test POST /join_room with valid room code"""
        # Create a room first
        create_response = self.app.post('/create_room', data={'nickname': 'Host'})
        
        # Extract room code from redirect location
        location = create_response.headers.get('Location')
        room_code = location.split('/room/')[1]
        
        # Join the room
        join_response = self.app.post('/join_room', data={
            'room_code': room_code,
            'nickname': 'Joiner'
        })
        
        self.assertEqual(join_response.status_code, 302)
        self.assertIn(f'/room/{room_code}', join_response.headers.get('Location'))
    
    def test_join_room_endpoint_invalid(self):
        """Test POST /join_room with invalid room code"""
        response = self.app.post('/join_room', data={
            'room_code': '9999',
            'nickname': 'TestPlayer'
        })
        
        # Should redirect back to home with error
        self.assertEqual(response.status_code, 302)
        self.assertIn('/', response.headers.get('Location'))
    
    def test_room_page_loads(self):
        """Test that room page loads for valid room code"""
        # Create a room first
        create_response = self.app.post('/create_room', data={'nickname': 'Host'})
        location = create_response.headers.get('Location')
        
        # Access the room page
        response = self.app.get(location)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Room', response.data)
    
    def test_room_page_invalid_code(self):
        """Test that room page returns 404 for invalid code"""
        response = self.app.get('/room/9999')
        self.assertEqual(response.status_code, 404)


class TestPlayerCountSelection(unittest.TestCase):
    """Test player count selection functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.rooms = {}
        self.players = {}
    
    def test_create_room_with_player_count(self):
        """Test that create_room accepts and stores max_players parameter"""
        from web.app import create_room, rooms
        
        nickname = "TestPlayer"
        max_players = 4
        room_code, session_id = create_room(nickname, max_players)
        
        # Room should store max_players
        self.assertIn(room_code, rooms)
        room = rooms[room_code]
        self.assertEqual(room['max_players'], max_players)
        self.assertEqual(len(room['players']), 1)  # Only creator initially
    
    def test_create_room_player_count_validation(self):
        """Test that create_room validates player count range (2-8)"""
        from web.app import create_room
        
        nickname = "TestPlayer"
        
        # Valid range should work
        for count in [2, 3, 4, 5, 6, 7, 8]:
            room_code, session_id = create_room(nickname, count)
            self.assertIsNotNone(room_code)
        
        # Invalid counts should raise ValueError
        with self.assertRaises(ValueError):
            create_room(nickname, 1)  # Too few
        with self.assertRaises(ValueError):
            create_room(nickname, 9)  # Too many
    
    def test_join_room_respects_max_players(self):
        """Test that rooms cannot exceed max_players limit"""
        from web.app import create_room, join_room, rooms
        
        # Create room with max 3 players
        host_nickname = "Host"
        max_players = 3
        room_code, host_session = create_room(host_nickname, max_players)
        
        # First joiner should succeed
        joiner1_session = join_room(room_code, "Joiner1")
        self.assertIsNotNone(joiner1_session)
        self.assertEqual(len(rooms[room_code]['players']), 2)
        
        # Second joiner should succeed (room full now)
        joiner2_session = join_room(room_code, "Joiner2")
        self.assertIsNotNone(joiner2_session)
        self.assertEqual(len(rooms[room_code]['players']), 3)
        
        # Third joiner should fail (room is full)
        joiner3_session = join_room(room_code, "Joiner3")
        self.assertIsNone(joiner3_session)
    
    def test_start_game_fills_with_ai_agents(self):
        """Test that start_game fills empty slots with AI agents"""
        from web.app import create_room, start_game, rooms
        
        # Create room with 4 max players but only 2 human players
        host_nickname = "Host"
        max_players = 4
        room_code, host_session = create_room(host_nickname, max_players)
        
        # Add one more human player
        from web.app import join_room
        joiner_session = join_room(room_code, "Joiner")
        
        # Start the game - should fill with 2 AI agents
        game_started = start_game(room_code, host_session)
        self.assertTrue(game_started)
        
        # Game state should be created
        room = rooms[room_code]
        self.assertIsNotNone(room['game_state'])
        
        # Should have 4 total players (2 human + 2 AI)
        game = room['game_state']
        self.assertEqual(game.num_players, 4)


class TestPlayerCountFlaskRoutes(unittest.TestCase):
    """Test Flask routes for player count selection"""
    
    def setUp(self):
        """Set up Flask test client"""
        from web.app import app
        self.app = app.test_client()
        self.app.testing = True
    
    def test_create_room_with_player_count_parameter(self):
        """Test POST /create_room with player_count parameter"""
        response = self.app.post('/create_room', data={
            'nickname': 'TestPlayer',
            'player_count': '5'
        })
        
        # Should redirect to room page
        self.assertEqual(response.status_code, 302)
        
        # Location header should contain room code
        location = response.headers.get('Location')
        self.assertIsNotNone(location)
        self.assertIn('/room/', location)
    
    def test_create_room_invalid_player_count(self):
        """Test POST /create_room with invalid player count"""
        # Test too low
        response = self.app.post('/create_room', data={
            'nickname': 'TestPlayer',
            'player_count': '1'
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn('/', response.headers.get('Location'))  # Redirect to home on error
        
        # Test too high
        response = self.app.post('/create_room', data={
            'nickname': 'TestPlayer',
            'player_count': '10'
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn('/', response.headers.get('Location'))  # Redirect to home on error
    
    def test_start_game_endpoint(self):
        """Test POST /start_game endpoint for room host"""
        # Create a room first
        create_response = self.app.post('/create_room', data={
            'nickname': 'Host',
            'player_count': '3'
        })
        
        # Extract room code from redirect location
        location = create_response.headers.get('Location')
        room_code = location.split('/room/')[1]
        
        # Start the game
        with self.app.session_transaction() as sess:
            # Simulate having a valid session
            pass
        
        start_response = self.app.post(f'/start_game/{room_code}')
        
        # Should redirect to game page or stay on room with game started
        self.assertEqual(start_response.status_code, 302)


if __name__ == '__main__':
    unittest.main()