"""
Interactive Perudo Game wrapper for web interface.
Provides turn-by-turn game state management using the existing PerudoSimulator.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.perudo import PerudoSimulator, Action


class InteractivePerudoGame:
    """
    Interactive wrapper around PerudoSimulator for turn-by-turn web gameplay.
    Maintains game state and provides methods for web interface.
    """
    
    def __init__(self, players, ai_agents=None, auto_continue_delay=3):
        """
        Initialize interactive game.
        
        Args:
            players: List of player names (human players)
            ai_agents: List of AI agent instances for remaining slots
            auto_continue_delay: Seconds to wait before auto-continuing after round end
        """
        self.players = players
        self.ai_agents = ai_agents or []
        self.total_players = len(players) + len(self.ai_agents)
        self.auto_continue_delay = auto_continue_delay
        self.auto_continue_timer = None
        
        # Initialize simulator
        self.simulator = PerudoSimulator(num_players=self.total_players)
        
        # Game state
        self.state = self.simulator.new_game()
        self.hands = None
        self.current_player = 0
        self.current_bid = None
        self.current_bid_maker = None
        self.game_over = False
        self.winner = None
        self.maputa_active = False
        self.maputa_restrict_face = None
        self.first_bid_by = None
        self.round_history = []  # Track actions within current round
        
        # Round end state management
        self.round_end_state = False  # True when showing round results
        self.round_end_loser = None
        self.round_end_last_bid = None
        self.round_end_actual_count = None
        self.round_end_all_hands = None
        self.human_continue_status = {}  # Track which humans have pressed continue
        
        # AI processing state management
        self.ai_thinking = False  # True when an AI is actively processing their turn
        self.ai_thinking_player = None  # Index of the AI player that is thinking
        
        # State change listeners for real-time updates
        self.state_listeners = []  # List of callback functions for state changes
        
        # Start new round
        self._start_new_round()
    
    def _start_new_round(self):
        """Start a new round with fresh dice rolls."""
        if self.is_game_over():
            return
            
        # Roll new hands for all players
        self.hands = self.simulator.roll_hands(self.state)
        
        # Find starting player (first alive player)
        alive_players = [i for i, c in enumerate(self.state['dice_counts']) if c > 0]
        if len(alive_players) <= 1:
            self.game_over = True
            self.winner = alive_players[0] if alive_players else None
            return
            
        # In Perudo, the player who lost a die starts the next round
        # BUT: if they're eliminated, the next alive player starts instead
        if (self.current_player >= self.total_players or self.current_player < 0 or 
            self.state['dice_counts'][self.current_player] == 0):
            # Find the next alive player starting from the loser's position
            start_search = self.current_player if (0 <= self.current_player < self.total_players) else 0
            for i in range(self.total_players):
                candidate = (start_search + i) % self.total_players
                if self.state['dice_counts'][candidate] > 0:
                    self.current_player = candidate
                    break
            else:
                # Fallback: use first alive player
                self.current_player = alive_players[0]
        
        # Check for maputa (single die rule)
        self.maputa_active = (self.simulator.use_maputa and 
                             self.state['dice_counts'][self.current_player] == 1)
        self.maputa_restrict_face = None
        self.current_bid = None
        self.current_bid_maker = None
        self.first_bid_by = None
        self.round_history = []  # Reset action history for new round
    
    def get_observation(self, player_index):
        """Get game observation for a specific player."""
        if self.is_game_over():
            return {
                'player_idx': player_index,
                'hand': [],
                'dice_counts': list(self.state['dice_counts']),
                'current_bid': self.current_bid,
                'legal_actions': [],
                'game_over': True,
                'winner': self.winner,
                'round_history': list(self.round_history)  # Include action history even when game over
            }
        
        # Get legal actions for current player
        legal_actions = []
        if player_index == self.current_player:
            legal_actions = self.simulator.legal_actions(
                self.state, 
                self.current_bid, 
                self.maputa_restrict_face
            )
        
        return {
            'player_idx': player_index,
            'hand': list(self.hands[player_index]) if self.hands else [],
            'dice_counts': list(self.state['dice_counts']),
            'current_bid': self.current_bid,
            'legal_actions': legal_actions,
            'game_over': False,
            'winner': None,
            'maputa_active': self.maputa_active,
            'maputa_restrict_face': self.maputa_restrict_face,
            'round_history': list(self.round_history),  # Include action history
            'ai_thinking': self.ai_thinking,  # Include AI thinking status
            'ai_thinking_player': self.ai_thinking_player  # Include which AI is thinking
        }
    
    def submit_bid(self, quantity, face):
        """Submit a bid action."""
        if self.is_game_over():
            raise ValueError("Game is over")
        
        action = Action.bid(quantity, face)
        return self._process_action(action)
    
    def submit_call(self):
        """Submit a call action."""
        if self.is_game_over():
            raise ValueError("Game is over")
        
        action = Action.call()
        return self._process_action(action)
    
    def submit_exact_call(self):
        """Submit an exact call action."""
        if self.is_game_over():
            raise ValueError("Game is over")
        
        action = Action.exact()
        return self._process_action(action)
    
    def _process_action(self, action):
        """Process a player action and update game state."""
        if self.is_game_over():
            return False
        
        # Validate action is legal
        legal_actions = self.simulator.legal_actions(
            self.state, 
            self.current_bid, 
            self.maputa_restrict_face
        )
        
        if action not in legal_actions:
            raise ValueError(f"Illegal action: {Action.to_str(action)}")
        
        # Record action in round history
        player_name = self.get_player_name(self.current_player)
        if action[0] == 'bid':
            action_desc = f"Bid {action[1]} × {action[2]}"
        elif action[0] == 'call':
            action_desc = "Call (Challenge)"
        elif action[0] == 'exact':
            action_desc = "Exact Call"
        else:
            action_desc = Action.to_str(action)
        
        self.round_history.append({
            'player': player_name,
            'action': action_desc,
            'player_index': self.current_player
        })
        
        if Action.is_bid(action) and self.first_bid_by is None:
            self.first_bid_by = self.current_player
            if self.maputa_active:
                self.maputa_restrict_face = Action.face(action)
        
        if action[0] == 'bid':
            # Process bid
            quantity, face = action[1], action[2]
            
            # Validate maputa restriction
            if (self.maputa_restrict_face is not None and 
                face != self.maputa_restrict_face):
                self._handle_round_end(self.current_player)
                return True
            
            self.current_bid = (quantity, face)
            self.current_bid_maker = self.current_player
            self.current_player = self.simulator.next_player_idx(
                self.current_player, self.state['dice_counts']
            )
            
        else:
            # Process call or exact
            if self.current_bid is None:
                self._handle_round_end(self.current_player)
                return True
            
            bid_true, actual_count = self.simulator.is_bid_true(
                self.hands, 
                self.current_bid,
                ones_are_wild=(self.simulator.ones_are_wild and not self.maputa_active)
            )
            
            if action[0] == 'call':
                loser = self.current_player if bid_true else self.current_bid_maker
                self._handle_round_end(loser)
                
            else:  # exact call
                if actual_count == self.current_bid[0]:
                    # Exact call succeeded - caller gains die, bid maker loses
                    self.state['dice_counts'][self.current_player] += 1
                    self._handle_round_end(self.current_bid_maker)
                else:
                    # Exact call failed - caller loses
                    self._handle_round_end(self.current_player)
        
            # Notify listeners of state change
            self._notify_state_change()
            return True
    
    def _handle_round_end(self, loser, actual_count=None):
        """Handle the end of a round when someone loses a die."""
        # Calculate actual count if not provided (for calls/exacts)
        if actual_count is None and self.current_bid:
            _, actual_count = self.simulator.is_bid_true(
                self.hands, 
                self.current_bid,
                ones_are_wild=(self.simulator.ones_are_wild and not self.maputa_active)
            )
        
        # Enter round end state
        self.round_end_state = True
        self.round_end_loser = loser
        self.round_end_last_bid = self.current_bid
        self.round_end_actual_count = actual_count
        self.round_end_all_hands = [list(hand) for hand in self.hands]  # Copy all hands
        
        # Initialize continue status for all human players
        self.human_continue_status = {}
        for i in range(len(self.players)):
            self.human_continue_status[i] = False
        
        # Set auto-continue timer if enabled
        if self.auto_continue_delay > 0:
            import threading
            import time
            def auto_continue():
                time.sleep(self.auto_continue_delay)
                if self.round_end_state:  # Still in round end state
                    self._continue_after_round_end()
            
            self.auto_continue_timer = threading.Thread(target=auto_continue)
            self.auto_continue_timer.daemon = True
            self.auto_continue_timer.start()
        
        # Remove a die from loser
        self.state['dice_counts'][loser] = max(0, self.state['dice_counts'][loser] - 1)
        
        # Check if game is over
        alive_players = [i for i, c in enumerate(self.state['dice_counts']) if c > 0]
        if len(alive_players) <= 1:
            self.game_over = True
            self.winner = alive_players[0] if alive_players else None
            self.round_end_state = False  # No need for round end state if game over
        else:
            # Will start new round after all humans continue
            self.current_player = loser
    
    def is_game_over(self):
        """Check if game is over."""
        return self.game_over
    
    def get_winner(self):
        """Get game winner (None if game not over)."""
        return self.winner if self.is_game_over() else None
    
    @property
    def dice_counts(self):
        """Get current dice counts for all players."""
        return list(self.state['dice_counts'])
    
    @property
    def num_players(self):
        """Get total number of players (backward compatibility)."""
        return self.total_players
    
    def get_player_count(self):
        """Get total number of players."""
        return self.total_players
    
    def is_human_player(self, player_index):
        """Check if a player index corresponds to a human player."""
        return player_index < len(self.players)
    
    def get_player_name(self, player_index):
        """Get player name for display."""
        if self.is_human_player(player_index):
            return self.players[player_index]
        else:
            ai_index = player_index - len(self.players)
            return f"AI_Bot_{ai_index + 1}"
    
    def submit_continue(self, player_index):
        """Submit continue action for a human player during round end state."""
        if not self.round_end_state:
            raise ValueError("Not in round end state")
        
        if not self.is_human_player(player_index):
            raise ValueError("Only human players can continue")
        
        # Mark this human player as having continued
        self.human_continue_status[player_index] = True
        
        # Check if all human players have continued
        if all(self.human_continue_status.values()):
            self._continue_after_round_end()
        
        return True
    
    def _continue_after_round_end(self):
        """Continue game after all human players have pressed continue."""
        if not self.round_end_state:
            return
        
        # Cancel auto-continue timer if it exists
        if self.auto_continue_timer and self.auto_continue_timer.is_alive():
            # Note: We can't directly cancel a sleeping thread, but the condition check
            # in auto_continue will prevent it from proceeding if round_end_state is False
            pass
        
        # Clear round end state
        self.round_end_state = False
        self.round_end_loser = None
        self.round_end_last_bid = None
        self.round_end_actual_count = None
        self.round_end_all_hands = None
        self.human_continue_status = {}
        self.auto_continue_timer = None
        
        # Start new round if game not over
        if not self.is_game_over():
            self._start_new_round()
            # Notify listeners of state change
            self._notify_state_change()
    
    def is_in_round_end_state(self):
        """Check if currently in round end state."""
        return self.round_end_state
    
    def get_round_end_info(self):
        """Get information about the round end for display."""
        if not self.round_end_state:
            return None
        
        return {
            'loser': self.round_end_loser,
            'loser_name': self.get_player_name(self.round_end_loser),
            'last_bid': self.round_end_last_bid,
            'actual_count': self.round_end_actual_count,
            'all_hands': self.round_end_all_hands,
            'continue_status': dict(self.human_continue_status)
        }
    
    def add_state_listener(self, callback):
        """Add callback to be notified of state changes."""
        self.state_listeners.append(callback)
    
    def _notify_state_change(self):
        """Notify all listeners of state change."""
        for callback in self.state_listeners:
            try:
                callback(self)
            except Exception as e:
                # Log error but don't break game
                pass
    
    def process_ai_turns(self):
        """Process all consecutive AI turns until it's a human player's turn or game ends."""
        import threading
        
        def process_ai_turns_async():
            """Process AI turns in background thread with immediate state updates."""
            while (not self.is_game_over() and 
                   not self.is_human_player(self.current_player)):
                
                ai_index = self.current_player - len(self.players)
                if ai_index >= len(self.ai_agents):
                    break
                
                # Set AI thinking status and notify immediately
                self.ai_thinking = True
                self.ai_thinking_player = self.current_player
                self._notify_state_change()
                
                try:
                    ai_agent = self.ai_agents[ai_index]
                    obs = self.get_observation(self.current_player)
                    
                    # Convert observation to format expected by agents
                    agent_obs = {
                        'player_idx': obs['player_idx'],
                        'my_hand': obs['hand'],
                        'dice_counts': obs['dice_counts'],
                        'current_bid': self.current_bid,
                        'history': [],
                        'maputa_active': obs.get('maputa_active', False),
                        'maputa_restrict_face': obs.get('maputa_restrict_face'),
                        '_simulator': self.simulator,
                    }
                    
                    action = ai_agent.select_action(agent_obs)
                    
                    if action[0] == 'bid':
                        self.submit_bid(action[1], action[2])
                    elif action[0] == 'call':
                        self.submit_call()
                    elif action[0] == 'exact':
                        self.submit_exact_call()
                        
                finally:
                    # Clear AI thinking status and notify
                    self.ai_thinking = False
                    self.ai_thinking_player = None
                    self._notify_state_change()
        
        # Run AI processing in background thread for non-blocking operation
        ai_thread = threading.Thread(target=process_ai_turns_async)
        ai_thread.daemon = True
        ai_thread.start()