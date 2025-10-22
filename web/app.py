"""
Flask web application for Perudo game - Week 1 MVP implementation.
Provides basic room creation and joining functionality.
"""

from flask import Flask, render_template, request, redirect, url_for, session, abort
import random
import uuid
import os
import sys

# Add parent directory to path for importing game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game modules
from sim.perudo import PerudoSimulator
from agents.mc_agent import MonteCarloAgent

app = Flask(__name__)
app.secret_key = 'perudo-mvp-secret-key-change-in-production'

# In-memory storage for MVP (as specified in requirements)
rooms = {}  # room_code: {'players': [], 'game_state': None, 'host': str}
players = {}  # session_id: {'nickname': str, 'room_code': str}


def create_room(nickname, max_players=4):
    """
    Create a new game room with a unique 4-digit code.
    
    Args:
        nickname (str): The nickname of the room creator
        max_players (int): Maximum number of players (2-8, default 4)
        
    Returns:
        tuple: (room_code, session_id) - 4-digit room code and 8-char session ID
        
    Raises:
        ValueError: If max_players is not in range 2-8
    """
    # Validate player count range
    if not isinstance(max_players, int) or max_players < 2 or max_players > 8:
        raise ValueError("max_players must be an integer between 2 and 8")
    
    # Generate unique 4-digit room code
    room_code = random.randint(1000, 9999)
    while room_code in rooms:
        room_code = random.randint(1000, 9999)
    
    # Generate 8-character session ID
    session_id = str(uuid.uuid4())[:8]
    
    # Store room data
    rooms[room_code] = {
        'players': [session_id],
        'game_state': None,
        'host': session_id,
        'max_players': max_players
    }
    
    # Store player data
    players[session_id] = {
        'nickname': nickname,
        'room_code': room_code
    }
    
    return room_code, session_id


def join_room(room_code, nickname):
    """
    Join an existing room with the given code.
    
    Args:
        room_code (int): The 4-digit room code
        nickname (str): The nickname of the joining player
        
    Returns:
        str or None: Session ID if successful, None if room doesn't exist or is full
    """
    if room_code not in rooms:
        return None
    
    room = rooms[room_code]
    
    # Check if room is full
    if len(room['players']) >= room['max_players']:
        return None
    
    # Generate session ID for the joining player
    session_id = str(uuid.uuid4())[:8]
    
    # Add player to room
    room['players'].append(session_id)
    
    # Store player data
    players[session_id] = {
        'nickname': nickname,
        'room_code': room_code
    }
    
    return session_id


def start_game(room_code, requesting_session_id):
    """
    Start a game in the given room, filling empty slots with AI agents.
    
    Args:
        room_code (int): The 4-digit room code
        requesting_session_id (str): Session ID of the player requesting to start
        
    Returns:
        bool: True if game started successfully, False otherwise
    """
    if room_code not in rooms:
        return False
    
    room = rooms[room_code]
    
    # Only the host can start the game
    if room['host'] != requesting_session_id:
        return False
    
    # Don't start if game already exists
    if room['game_state'] is not None:
        return False
    
    # Get human player count
    num_human_players = len(room['players'])
    
    # Create AI agents for empty slots
    ai_agents = []
    slots_to_fill = room['max_players'] - num_human_players
    for i in range(slots_to_fill):
        ai_name = f"AI_Bot_{i+1}"
        ai_agent = MonteCarloAgent(
            name=ai_name,
            n=100,  # Reduced simulations for web performance
            chunk_size=4,
            max_rounds=4,
            early_stop_margin=0.15
        )
        ai_agents.append(ai_agent)
    
    # Create game instance with total player count
    game = PerudoSimulator(num_players=room['max_players'])
    
    # Store game state and AI agents
    room['game_state'] = game
    room['ai_agents'] = ai_agents
    
    return True


@app.route('/')
def home():
    """Home page with room creation and joining forms."""
    return render_template('index.html')


@app.route('/create_room', methods=['POST'])
def create_room_endpoint():
    """Handle room creation form submission."""
    nickname = request.form.get('nickname', '').strip()
    player_count_str = request.form.get('player_count', '4').strip()
    
    # Basic validation
    if not nickname or len(nickname) < 2:
        return redirect(url_for('home'))
    
    # Validate player count
    try:
        player_count = int(player_count_str)
    except (ValueError, TypeError):
        return redirect(url_for('home'))
    
    # Create room and get session
    try:
        room_code, session_id = create_room(nickname, player_count)
    except ValueError:
        # Invalid player count range (not 2-8)
        return redirect(url_for('home'))
    
    # Store session ID in Flask session
    session['session_id'] = session_id
    
    # Redirect to room page
    return redirect(url_for('room_page', code=room_code))


@app.route('/join_room', methods=['POST'])
def join_room_endpoint():
    """Handle room joining form submission."""
    room_code_str = request.form.get('room_code', '').strip()
    nickname = request.form.get('nickname', '').strip()
    
    # Basic validation
    if not room_code_str or not nickname or len(nickname) < 2:
        return redirect(url_for('home'))
    
    try:
        room_code = int(room_code_str)
    except ValueError:
        return redirect(url_for('home'))
    
    # Try to join room
    session_id = join_room(room_code, nickname)
    
    if session_id is None:
        # Room doesn't exist, redirect back to home
        return redirect(url_for('home'))
    
    # Store session ID in Flask session
    session['session_id'] = session_id
    
    # Redirect to room page
    return redirect(url_for('room_page', code=room_code))


@app.route('/room/<int:code>')
def room_page(code):
    """Display room lobby page."""
    if code not in rooms:
        abort(404)
    
    # Get current session
    session_id = session.get('session_id')
    if session_id not in players:
        # Invalid session, redirect to home
        return redirect(url_for('home'))
    
    # Check if player belongs to this room
    if players[session_id]['room_code'] != code:
        # Player doesn't belong to this room
        abort(404)
    
    room_data = rooms[code]
    room_players = []
    
    # Get player nicknames
    for player_id in room_data['players']:
        if player_id in players:
            room_players.append({
                'nickname': players[player_id]['nickname'],
                'is_host': player_id == room_data['host'],
                'session_id': player_id
            })
    
    return render_template('room.html', 
                         room_code=code, 
                         players=room_players,
                         current_player=session_id,
                         max_players=room_data['max_players'])


@app.route('/start_game/<int:code>', methods=['POST'])
def start_game_endpoint(code):
    """Handle game start request from room host."""
    if code not in rooms:
        abort(404)
    
    # Get current session
    session_id = session.get('session_id')
    if session_id not in players:
        # Invalid session, redirect to home
        return redirect(url_for('home'))
    
    # Check if player belongs to this room
    if players[session_id]['room_code'] != code:
        # Player doesn't belong to this room
        abort(404)
    
    # Try to start the game
    game_started = start_game(code, session_id)
    
    if game_started:
        # Game started successfully, stay on room page (game will be visible)
        return redirect(url_for('room_page', code=code))
    else:
        # Failed to start (not host, already started, etc.)
        return redirect(url_for('room_page', code=code))


if __name__ == '__main__':
    # Run development server
    app.run(debug=True, host='127.0.0.1', port=5000)