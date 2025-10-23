"""
Flask web application for Perudo game - Week 1 MVP implementation.
Provides basic room creation and joining functionality.
"""

from flask import Flask, render_template, request, redirect, url_for, session, abort, jsonify, Response
import random
import uuid
import os
import sys
import json
import time
import threading

# Add parent directory to path for importing game modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game modules
from sim.perudo import PerudoSimulator
from agents.random_agent import RandomAgent
from agents.baseline_agent import BaselineAgent
from agents.mc_agent import MonteCarloAgent
from web.interactive_game import InteractivePerudoGame

app = Flask(__name__)
app.secret_key = 'perudo-mvp-secret-key-change-in-production'

# In-memory storage for MVP (as specified in requirements)
rooms = {}  # room_code: {'players': [], 'game_state': None, 'host': str, 'ai_configs': []}
players = {}  # session_id: {'nickname': str, 'room_code': str}


def load_mc_config():
    """Load MonteCarloAgent configuration from mc_config.json."""
    try:
        config_path = os.path.join(os.path.dirname(__file__), 'mc_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        # Return default configuration if file not found or invalid
        print(f"Warning: Could not load mc_config.json: {e}")
        return {
            "name": "MC_Agent",
            "n": 100,
            "chunk_size": 8,
            "max_rounds": 4,
            "simulate_to_round_end": True,
            "early_stop_margin": 0.15,
            "weighted_sampling": False,
            "enable_parallel": False,
            "num_workers": None,
            "enhanced_pruning": False,
            "variance_reduction": False,
            "betting_history_enabled": False,
            "player_trust_enabled": False,
            "trust_learning_rate": 0.1,
            "history_memory_rounds": 10,
            "bayesian_sampling": False
        }


def create_agent_from_config(agent_config):
    """Create an AI agent instance from configuration."""
    agent_type = agent_config.get('type', 'random')
    agent_name = agent_config.get('name', 'AI_Agent')
    
    if agent_type == 'random':
        return RandomAgent(name=agent_name)
    elif agent_type == 'baseline':
        return BaselineAgent(name=agent_name)
    elif agent_type == 'mc':
        # Load MC configuration and create agent
        mc_config = load_mc_config()
        # Remove 'name' and documentation fields from mc_config as they conflict with or are not constructor parameters
        mc_params = {k: v for k, v in mc_config.items() if k != 'name' and not k.startswith('_')}
        return MonteCarloAgent(name=agent_name, **mc_params)
    else:
        # Default to random agent for unknown types
        return RandomAgent(name=agent_name)


def create_room(nickname, max_players=4, ai_configs=None):
    """
    Create a new game room with a unique 4-digit code, pre-populated with AI players.
    
    Args:
        nickname (str): The nickname of the room creator
        max_players (int): Maximum number of players (2-8, default 4)
        ai_configs (list): List of AI configurations for non-human slots
        
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
    
    # Create default AI configurations if none provided
    if ai_configs is None:
        ai_configs = []
        for i in range(max_players - 1):  # -1 for the human host
            ai_configs.append({
                'type': 'random',
                'name': f'AI_Bot_{i+1}',
                'is_ai': True
            })
    
    # Create player list with human host first, then AI placeholders
    players_list = [{'session_id': session_id, 'is_ai': False}]
    for ai_config in ai_configs:
        ai_session_id = f"ai_{uuid.uuid4().hex[:8]}"
        players_list.append({
            'session_id': ai_session_id, 
            'is_ai': True,
            'ai_config': ai_config
        })
    
    # Store room data
    rooms[room_code] = {
        'players': players_list,
        'game_state': None,
        'host': session_id,
        'max_players': max_players,
        'ai_configs': ai_configs
    }
    
    # Store human player data
    players[session_id] = {
        'nickname': nickname,
        'room_code': room_code,
        'is_ai': False
    }
    
    # Store AI player data
    for player in players_list[1:]:  # Skip human host
        if player['is_ai']:
            players[player['session_id']] = {
                'nickname': player['ai_config']['name'],
                'room_code': room_code,
                'is_ai': True,
                'ai_config': player['ai_config']
            }
    
    return room_code, session_id


def join_room(room_code, nickname):
    """
    Join an existing room by replacing the topmost AI player.
    
    Args:
        room_code (int): The 4-digit room code
        nickname (str): The nickname of the joining player
        
    Returns:
        str or None: Session ID if successful, None if room doesn't exist or no AI slots available
    """
    if room_code not in rooms:
        return None
    
    room = rooms[room_code]
    
    # Find the first AI player to replace (search from top to bottom)
    ai_player_index = None
    for i, player in enumerate(room['players']):
        if player['is_ai']:
            ai_player_index = i
            break
    
    # No AI players to replace - room is full of humans
    if ai_player_index is None:
        return None
    
    # Generate session ID for the joining player
    session_id = str(uuid.uuid4())[:8]
    
    # Get the AI player being replaced
    ai_player = room['players'][ai_player_index]
    old_ai_session_id = ai_player['session_id']
    
    # Remove old AI player data
    if old_ai_session_id in players:
        del players[old_ai_session_id]
    
    # Replace AI player with human player
    room['players'][ai_player_index] = {
        'session_id': session_id,
        'is_ai': False
    }
    
    # Store new human player data
    players[session_id] = {
        'nickname': nickname,
        'room_code': room_code,
        'is_ai': False
    }
    
    return session_id


def start_game(room_code, requesting_session_id):
    """
    Start a game in the given room using pre-configured players and AI agents.
    
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
    
    # Separate human and AI players from the room's player list
    human_players = []
    ai_agents = []
    
    for player_entry in room['players']:
        player_session_id = player_entry['session_id']
        
        if player_session_id in players:
            player_data = players[player_session_id]
            
            if player_data.get('is_ai', False):
                # Create AI agent from configuration
                ai_config = player_data.get('ai_config', {'type': 'random', 'name': 'AI_Agent'})
                ai_agent = create_agent_from_config(ai_config)
                ai_agents.append(ai_agent)
            else:
                # Human player
                human_players.append(player_data['nickname'])
    
    # Create interactive game instance with ordered players
    game = InteractivePerudoGame(human_players, ai_agents)
    
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
    """Handle room creation form submission with AI configuration."""
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
    
    # Parse AI configurations
    ai_configs = []
    for i in range(1, player_count):  # AI slots = player_count - 1 (human host)
        ai_type = request.form.get(f'ai_{i}_type', 'random')
        ai_configs.append({
            'type': ai_type,
            'name': f'AI_Bot_{i}',
            'is_ai': True
        })
    
    # Create room with AI configurations
    try:
        room_code, session_id = create_room(nickname, player_count, ai_configs)
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
    
    # Get player information from the new room structure
    for player_entry in room_data['players']:
        player_session_id = player_entry['session_id']
        if player_session_id in players:
            player_data = players[player_session_id]
            room_players.append({
                'nickname': player_data['nickname'],
                'is_host': player_session_id == room_data['host'],
                'session_id': player_session_id,
                'is_ai': player_data.get('is_ai', False),
                'agent_type': player_data.get('ai_config', {}).get('type', 'human') if player_data.get('is_ai', False) else 'human'
            })
    
    return render_template('room.html', 
                         room_code=code, 
                         players=room_players,
                         current_player=session_id,
                         max_players=room_data['max_players'],
                         game_started=room_data['game_state'] is not None)


@app.route('/game/<int:code>')
def game_page(code):
    """Display game interface page."""
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
    
    # Check if game has started
    if room_data['game_state'] is None:
        # Game not started, redirect to room lobby
        return redirect(url_for('room_page', code=code))
    
    return render_template('game.html', 
                         room_code=code,
                         session_id=session_id)


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
        # Game started successfully, redirect to game page
        return redirect(url_for('game_page', code=code))
    else:
        # Failed to start (not host, already started, etc.)
        return redirect(url_for('room_page', code=code))


@app.route('/action', methods=['POST'])
def submit_action():
    """Handle game action submission from players."""
    # Handle both JSON and form data
    if request.is_json:
        data = request.get_json()
    else:
        data = request.form.to_dict()
    
    if not data or 'room_code' not in data or 'action_type' not in data:
        return jsonify({'error': 'Invalid request data'}), 400
    
    try:
        room_code = int(data['room_code'])
    except (ValueError, TypeError):
        return jsonify({'error': 'Invalid room code'}), 400
    
    action_type = data['action_type']
    
    # Validate room exists
    if room_code not in rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room_data = rooms[room_code]
    game = room_data.get('game_state')
    
    if game is None:
        return jsonify({'error': 'Game not started'}), 400
    
    # Get current session
    session_id = session.get('session_id')
    if session_id not in players:
        return jsonify({'error': 'Invalid session'}), 401
    
    # Check if player belongs to this room
    if players[session_id]['room_code'] != room_code:
        return jsonify({'error': 'Player not in room'}), 403
    
    # Find player index in game
    player_nicknames = []
    for player_entry in room_data['players']:
        player_session_id = player_entry['session_id']
        if player_session_id in players:
            player_nicknames.append(players[player_session_id]['nickname'])
    
    try:
        player_index = player_nicknames.index(players[session_id]['nickname'])
    except ValueError:
        return jsonify({'error': 'Player not found in game'}), 400
    
    # Check if it's the player's turn (except for continue actions during round end)
    if action_type != 'continue' and game.current_player != player_index:
        return jsonify({'error': 'Not your turn'}), 400
    
    # Special validation for continue actions
    if action_type == 'continue' and not game.is_in_round_end_state():
        return jsonify({'error': 'Can only continue during round end'}), 400
    
    try:
        # Process action based on type
        if action_type == 'bid':
            if 'quantity' not in data or 'face' not in data:
                return jsonify({'error': 'Missing bid parameters'}), 400
            
            quantity = int(data['quantity'])
            face = int(data['face'])
            game.submit_bid(quantity, face)
            
        elif action_type == 'call':
            game.submit_call()
            
        elif action_type == 'exact':
            game.submit_exact_call()
            
        elif action_type == 'continue':
            game.submit_continue(player_index)
            
        else:
            return jsonify({'error': 'Invalid action type'}), 400
    
    except Exception as e:
        return jsonify({'error': f'Action failed: {str(e)}'}), 400
    
    # Process bot turns after human action - BUT NOT during round end states
    if not game.is_in_round_end_state():
        game.process_ai_turns()
    
    return jsonify({'success': True, 'game_over': game.is_game_over()})


@app.route('/poll/<int:code>')
def poll_game_state(code):
    """Get current game state for polling."""
    if code not in rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room_data = rooms[code]
    game = room_data.get('game_state')
    
    if game is None:
        return jsonify({'error': 'Game not started'}), 400
    
    # Get current session
    session_id = session.get('session_id')
    if session_id not in players:
        return jsonify({'error': 'Invalid session'}), 401
    
    # Check if player belongs to this room
    if players[session_id]['room_code'] != code:
        return jsonify({'error': 'Player not in room'}), 403
    
    # Find player index
    player_nicknames = []
    for player_entry in room_data['players']:
        player_session_id = player_entry['session_id']
        if player_session_id in players:
            player_nicknames.append(players[player_session_id]['nickname'])
    
    try:
        player_index = player_nicknames.index(players[session_id]['nickname'])
    except ValueError:
        return jsonify({'error': 'Player not found in game'}), 400
    
    # Get game observation for this player
    obs = game.get_observation(player_index)
    
    # Build response with game state
    response = {
        'current_player': game.current_player,
        'is_your_turn': game.current_player == player_index,
        'your_player_index': player_index,  # Add current user's player index
        'your_dice': obs['hand'],
        'current_bid': {
            'quantity': obs['current_bid'][0] if obs['current_bid'] else None,
            'face': obs['current_bid'][1] if obs['current_bid'] else None
        } if obs['current_bid'] else None,
        'legal_actions': [
            {
                'type': action[0],
                'quantity': action[1] if len(action) > 1 else None,
                'face': action[2] if len(action) > 2 else None
            }
            for action in obs['legal_actions']
        ],
        'game_over': game.is_game_over(),
        'winner': game.get_winner() if game.is_game_over() else None,
        'players': [
            {
                'name': game.get_player_name(i),
                'dice_count': game.dice_counts[i],
                'is_ai': not game.is_human_player(i)
            }
            for i in range(game.get_player_count())
        ],
        'round_history': obs.get('round_history', []),  # Include action history
        'round_end_state': game.is_in_round_end_state(),
        'round_end_info': game.get_round_end_info(),
        'ai_thinking': obs.get('ai_thinking', False),  # Include AI thinking status
        'ai_thinking_player': obs.get('ai_thinking_player', None)  # Include which AI is thinking
    }
    
    return jsonify(response)


@app.route('/stream/<int:code>')
def stream_game_state(code):
    """Stream game state updates using Server-Sent Events."""
    if code not in rooms:
        return jsonify({'error': 'Room not found'}), 404
    
    room_data = rooms[code]
    game = room_data.get('game_state')
    
    if game is None:
        return jsonify({'error': 'Game not started'}), 400
    
    # Get current session
    session_id = session.get('session_id')
    if session_id not in players:
        return jsonify({'error': 'Invalid session'}), 401
    
    # Check if player belongs to this room
    if players[session_id]['room_code'] != code:
        return jsonify({'error': 'Player not in room'}), 403
    
    # Find player index
    player_nicknames = []
    for player_entry in room_data['players']:
        player_session_id = player_entry['session_id']
        if player_session_id in players:
            player_nicknames.append(players[player_session_id]['nickname'])
    
    try:
        player_index = player_nicknames.index(players[session_id]['nickname'])
    except ValueError:
        return jsonify({'error': 'Player not found in game'}), 400
    
    def generate_updates():
        """Generate SSE updates for game state changes."""
        last_state = None
        state_changed = threading.Event()
        listener_added = False
        
        # Add state change listener to game
        def on_state_change(game_instance):
            state_changed.set()
        
        try:
            game.add_state_listener(on_state_change)
            listener_added = True
            
            # Send initial state immediately
            def build_current_state():
                obs = game.get_observation(player_index)
                return {
                    'current_player': game.current_player,
                    'is_your_turn': game.current_player == player_index,
                    'your_player_index': player_index,
                    'your_dice': obs['hand'],
                    'current_bid': {
                        'quantity': obs['current_bid'][0] if obs['current_bid'] else None,
                        'face': obs['current_bid'][1] if obs['current_bid'] else None
                    } if obs['current_bid'] else None,
                    'legal_actions': [
                        {
                            'type': action[0],
                            'quantity': action[1] if len(action) > 1 else None,
                            'face': action[2] if len(action) > 2 else None
                        }
                        for action in obs['legal_actions']
                    ],
                    'game_over': game.is_game_over(),
                    'winner': game.get_winner() if game.is_game_over() else None,
                    'players': [
                        {
                            'name': game.get_player_name(i),
                            'dice_count': game.dice_counts[i],
                            'is_ai': not game.is_human_player(i)
                        }
                        for i in range(game.get_player_count())
                    ],
                    'round_history': obs.get('round_history', []),
                    'round_end_state': game.is_in_round_end_state(),
                    'round_end_info': game.get_round_end_info(),
                    'ai_thinking': obs.get('ai_thinking', False),
                    'ai_thinking_player': obs.get('ai_thinking_player', None)
                }
            
            # Send initial state
            current_state = build_current_state()
            yield f"data: {json.dumps(current_state)}\n\n"
            last_state = current_state.copy()
            
            # If game is already over, close immediately
            if current_state['game_over']:
                return
            
            while True:
                try:
                    # Check if room/game still exists
                    if code not in rooms or rooms[code].get('game_state') != game:
                        break
                    
                    # Wait for state change with timeout
                    if state_changed.wait(timeout=30):  # 30 second timeout for keepalive
                        state_changed.clear()  # Reset the event
                        
                        # Get updated state
                        current_state = build_current_state()
                        
                        # Send update if state actually changed
                        if current_state != last_state:
                            yield f"data: {json.dumps(current_state)}\n\n"
                            last_state = current_state.copy()
                            
                            # If game is over, close the stream
                            if current_state['game_over']:
                                break
                    else:
                        # Timeout reached - send keepalive heartbeat
                        yield f": keepalive\n\n"
                        
                except Exception as e:
                    # Send error and close stream
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    break
                    
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Stream initialization failed: {str(e)}'})}\n\n"
        finally:
            # Clean up listener if it was added
            if listener_added:
                try:
                    # Note: We can't easily remove listeners without modifying InteractivePerudoGame
                    # For now, the listener will remain but become inactive when this function ends
                    pass
                except:
                    pass
    
    # Create response with proper SSE headers
    response = Response(generate_updates(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == '__main__':
    # Run development server
    app.run(debug=True, host='127.0.0.1', port=5000)