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
    
    def __init__(self, players, ai_agents=None):
        """
        Initialize interactive game.
        
        Args:
            players: List of player names (human players)
            ai_agents: List of AI agent instances for remaining slots
        """
        self.players = players
        self.ai_agents = ai_agents or []
        self.total_players = len(players) + len(self.ai_agents)
        
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
            
        # Find valid starting player
        while self.current_player not in alive_players:
            self.current_player = self.simulator.next_player_idx(self.current_player, self.state['dice_counts'])
        
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
            'round_history': list(self.round_history)  # Include action history
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
        
        return True
    
    def _handle_round_end(self, loser):
        """Handle the end of a round when someone loses a die."""
        # Remove a die from loser
        self.state['dice_counts'][loser] = max(0, self.state['dice_counts'][loser] - 1)
        
        # Check if game is over
        alive_players = [i for i, c in enumerate(self.state['dice_counts']) if c > 0]
        if len(alive_players) <= 1:
            self.game_over = True
            self.winner = alive_players[0] if alive_players else None
        else:
            # Start new round with loser as starting player
            self.current_player = loser
            self._start_new_round()
    
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
    
    def process_ai_turns(self):
        """Process all consecutive AI turns until it's a human player's turn or game ends."""
        while (not self.is_game_over() and 
               not self.is_human_player(self.current_player)):
            
            ai_index = self.current_player - len(self.players)
            if ai_index >= len(self.ai_agents):
                break
            
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