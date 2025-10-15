"""
Monte Carlo Agent Utilities

This module contains helper functions for the MonteCarloAgent including:
- Weighted determinization sampling based on bidding history
- Enhanced action scoring for advanced pruning
- Control variate baseline calculations for variance reduction
- Heuristic win probability estimation
- Betting history tracking and analysis
- Bayesian player modeling and trust management
"""

import random
import math
from sim.perudo import binom_cdf_ge_fast


class BettingHistoryEntry:
    """Single betting action with context."""
    
    def __init__(self, player_idx, action, round_num, dice_count, 
                 actual_hand=None, bid_result=None):
        self.player_idx = player_idx        # Who made the bid
        self.action = action                # ('bid', qty, face) or ('call',) or ('exact',)
        self.round_num = round_num          # Which round this occurred in
        self.dice_count = dice_count        # How many dice this player had
        self.actual_hand = actual_hand      # Player's actual dice (when revealed)
        self.bid_result = bid_result        # 'won_round', 'lost_dice', 'gained_dice', None


class GameBettingHistory:
    """Complete game betting history with analysis capabilities."""
    
    def __init__(self, num_players):
        self.num_players = num_players
        self.entries = []                   # All betting entries chronologically
        self.player_entries = [[] for _ in range(num_players)]  # Per-player entries
        self.round_entries = []             # Entries grouped by round
        self.current_round = 0
        
    def add_entry(self, entry):
        """Add a new betting entry to the history."""
        self.entries.append(entry)
        self.player_entries[entry.player_idx].append(entry)
        
        # Ensure we have enough round lists
        while len(self.round_entries) <= entry.round_num:
            self.round_entries.append([])
        
        self.round_entries[entry.round_num].append(entry)
        
    def get_player_bids(self, player_idx, face=None):
        """Get all bids made by a specific player, optionally filtered by face."""
        bids = [entry for entry in self.player_entries[player_idx] 
                if entry.action[0] == 'bid']
        
        if face is not None:
            bids = [bid for bid in bids if bid.action[2] == face]
        
        return bids
        
    def get_face_popularity(self, face, recent_rounds=None):
        """Calculate how frequently a face has been bid on by all players."""
        if recent_rounds is None:
            # Consider all rounds
            relevant_entries = self.entries
        else:
            # Consider only recent rounds
            min_round = max(0, self.current_round - recent_rounds + 1)
            relevant_entries = [entry for entry in self.entries 
                              if entry.round_num >= min_round]
        
        if not relevant_entries:
            return 1.0 / 6.0  # Default uniform probability
        
        # Count bids on this face vs total bids
        face_bids = sum(1 for entry in relevant_entries 
                       if entry.action[0] == 'bid' and entry.action[2] == face)
        total_bids = sum(1 for entry in relevant_entries if entry.action[0] == 'bid')
        
        if total_bids == 0:
            return 1.0 / 6.0  # Default uniform probability
        
        return face_bids / total_bids
        
    def get_round_entries(self, round_offset=-1):
        """Get entries from a specific round (default: last round)."""
        if round_offset < 0:
            target_round = len(self.round_entries) + round_offset
        else:
            target_round = round_offset
            
        if 0 <= target_round < len(self.round_entries):
            return self.round_entries[target_round]
        return []
        
    def analyze_player_accuracy(self, player_idx):
        """Analyze how accurate a player's bids have been historically."""
        player_bids = self.get_player_bids(player_idx)
        
        if not player_bids:
            return 0.5  # Default neutral accuracy
        
        total_accuracy = 0.0
        evaluated_bids = 0
        
        for bid_entry in player_bids:
            if bid_entry.actual_hand is not None:
                bid_face = bid_entry.action[2]
                actual_face_count = sum(1 for d in bid_entry.actual_hand 
                                      if d == bid_face or d == 1)  # Include ones as wild
                expected_proportion = actual_face_count / len(bid_entry.actual_hand)
                
                # Higher accuracy for players whose bids correlate with their actual dice
                accuracy = min(1.0, expected_proportion * 2)  # Scale to reasonable range
                total_accuracy += accuracy
                evaluated_bids += 1
        
        if evaluated_bids == 0:
            return 0.5  # Default neutral accuracy
        
        return total_accuracy / evaluated_bids


class PlayerTrustManager:
    """Manages dynamic trust parameters for all players."""
    
    def __init__(self, num_players, initial_trust=0.5):
        self.trust_params = [initial_trust] * num_players
        self.accuracy_history = [[] for _ in range(num_players)]
        
    def update_trust_after_round(self, round_result, betting_history):
        """Update trust parameters based on round outcome."""
        for player_idx in range(len(self.trust_params)):
            # Calculate bidding accuracy for this round
            accuracy = self.calculate_round_accuracy(player_idx, round_result, betting_history)
            
            if accuracy is not None:
                self.accuracy_history[player_idx].append(accuracy)
                
                # Update trust using exponential moving average
                current_trust = self.trust_params[player_idx]
                learning_rate = 0.1
                
                # Accuracy > 0.5 increases trust, < 0.5 decreases trust
                trust_delta = (accuracy - 0.5) * learning_rate
                new_trust = current_trust + trust_delta
                
                # Keep trust in valid range [0, 1]
                self.trust_params[player_idx] = max(0.0, min(1.0, new_trust))
    
    def calculate_round_accuracy(self, player_idx, round_result, betting_history):
        """Calculate how accurate a player's bids were this round."""
        player_bids = [entry for entry in betting_history.get_round_entries(-1) 
                      if entry.player_idx == player_idx and entry.action[0] == 'bid']
        
        if not player_bids:
            return None  # No bids to evaluate
        
        # Calculate correlation between bids and actual dice
        total_accuracy = 0.0
        for bid_entry in player_bids:
            if bid_entry.actual_hand is not None:
                bid_face = bid_entry.action[2]
                actual_face_count = sum(1 for d in bid_entry.actual_hand 
                                      if d == bid_face or (d == 1 and bid_face != 1))
                expected_proportion = actual_face_count / len(bid_entry.actual_hand)
                
                # Higher accuracy for players whose bids correlate with their actual dice
                base_accuracy = min(1.0, expected_proportion * 2)  # Scale to reasonable range
                total_accuracy += base_accuracy
        
        return total_accuracy / len(player_bids)
    
    def get_trust(self, player_idx):
        """Get current trust level for a player."""
        if 0 <= player_idx < len(self.trust_params):
            return self.trust_params[player_idx]
        return 0.5  # Default neutral trust


def sample_weighted_dice(agent, obs, num_dice):
    """Sample dice with weights based on bidding history plausibility."""
    current_bid = obs.get('current_bid')
    sim = obs['_simulator']
    
    # If no current bid, fall back to uniform sampling
    if current_bid is None:
        return [agent.rng.randint(1, 6) for _ in range(num_dice)]
    
    bid_qty, bid_face = current_bid
    
    # Calculate face probabilities based on bid plausibility
    # Higher weight for faces that make the current bid more plausible
    face_weights = [1.0] * 6  # Base weights for faces 1-6
    
    # If ones are wild and not in maputa, faces other than bid_face get boost from ones
    ones_wild = sim.ones_are_wild and not obs.get('maputa_active', False)
    
    if bid_face == 1:
        # Bid is for ones - increase weight for ones
        face_weights[0] = 2.0  # Face 1 gets higher weight
    else:
        # Bid is for non-ones
        if ones_wild:
            # Both bid_face and ones contribute to the bid
            face_weights[0] = 1.5  # Ones get moderate boost (wild)
            face_weights[bid_face - 1] = 2.0  # Bid face gets higher weight
        else:
            # Only bid_face contributes
            face_weights[bid_face - 1] = 2.0  # Bid face gets higher weight
    
    # Calculate total remaining dice needed to make bid plausible
    my_hand = obs['my_hand']
    my_contribution = sum(1 for d in my_hand if d == bid_face or (ones_wild and d == 1 and bid_face != 1))
    total_dice_needed = max(0, bid_qty - my_contribution)
    
    # Adjust weights based on how many dice we need
    # More aggressive weighting if we need many dice to make bid true
    if total_dice_needed > 0:
        total_other_dice = sum(obs['dice_counts']) - len(my_hand)
        if total_other_dice > 0:
            needed_ratio = min(1.0, total_dice_needed / total_other_dice)
            # Scale weights more aggressively for higher ratios
            scaling_factor = 1.0 + needed_ratio * 2.0
            if bid_face == 1:
                face_weights[0] *= scaling_factor
            else:
                if ones_wild:
                    face_weights[0] *= (1.0 + needed_ratio * 1.0)  # Ones get moderate boost
                face_weights[bid_face - 1] *= scaling_factor
    
    # Normalize weights
    total_weight = sum(face_weights)
    face_probs = [w / total_weight for w in face_weights]
    
    # Sample dice using weighted probabilities
    dice = []
    for _ in range(num_dice):
        # Use weighted random choice
        r = agent.rng.random()
        cumsum = 0.0
        for face, prob in enumerate(face_probs, 1):
            cumsum += prob
            if r <= cumsum:
                dice.append(face)
                break
        else:
            # Fallback (should rarely happen)
            dice.append(6)
    
    return dice


def calculate_recency_weight(bid_round, current_round, decay_factor=0.8):
    """Calculate recency weight for historical bids."""
    rounds_ago = current_round - bid_round
    return max(0.1, decay_factor ** rounds_ago)  # Minimum weight of 0.1


def calculate_trust_weighted_accuracy(history, trust_params, face):
    """Calculate trust-weighted accuracy for bids on a specific face."""
    if not history.entries:
        return 0.5  # Default neutral accuracy
    
    total_weighted_accuracy = 0.0
    total_weight = 0.0
    
    for entry in history.entries:
        if entry.action[0] == 'bid' and entry.action[2] == face:
            if entry.actual_hand is not None:
                player_idx = entry.player_idx
                trust = trust_params[player_idx] if player_idx < len(trust_params) else 0.5
                
                # Calculate accuracy for this bid
                bid_face = entry.action[2]
                actual_face_count = sum(1 for d in entry.actual_hand 
                                      if d == bid_face or (d == 1 and bid_face != 1))
                expected_proportion = actual_face_count / len(entry.actual_hand)
                accuracy = min(1.0, expected_proportion * 2)  # Scale to reasonable range
                
                # Weight by trust and recency
                recency_weight = calculate_recency_weight(entry.round_num, history.current_round)
                weight = trust * recency_weight
                
                total_weighted_accuracy += accuracy * weight
                total_weight += weight
    
    if total_weight == 0:
        return 0.5  # Default neutral accuracy
    
    return total_weighted_accuracy / total_weight


def compute_collective_plausibility(obs, bid_qty, bid_face):
    """
    Calculate bid plausibility based on collective betting patterns.
    
    Factors considered:
    1. How often this face has been bid on by all players
    2. Historical accuracy of bids on this face
    3. Current game state vs historical patterns
    4. Player trust levels for those who have bid this face
    """
    # Check if betting history is available
    history = obs.get('betting_history')
    trust_params = obs.get('player_trust', [])
    
    if history is None:
        # Fall back to basic probability calculation
        return calculate_base_probability(obs, bid_qty, bid_face)
    
    # Base statistical probability
    base_prob = calculate_base_probability(obs, bid_qty, bid_face)
    
    # Historical face popularity adjustment
    face_popularity = history.get_face_popularity(bid_face, recent_rounds=3)
    popularity_modifier = 1.0 + (face_popularity - 0.167) * 0.5  # 0.167 = 1/6 baseline
    
    # Trust-weighted historical accuracy
    face_accuracy = calculate_trust_weighted_accuracy(history, trust_params, bid_face)
    accuracy_modifier = 0.8 + (face_accuracy * 0.4)  # Range: 0.8 to 1.2
    
    # Combined plausibility score
    plausibility = base_prob * popularity_modifier * accuracy_modifier
    
    return max(0.0, min(1.0, plausibility))


def calculate_base_probability(obs, bid_qty, bid_face):
    """Calculate base statistical probability for a bid."""
    dice_counts = obs['dice_counts']
    my_hand = obs['my_hand']
    sim = obs['_simulator']
    
    total_dice = sum(dice_counts)
    my_dice_count = len(my_hand)
    opponent_dice = total_dice - my_dice_count
    
    if opponent_dice <= 0:
        return 1.0 if bid_qty <= 0 else 0.0
    
    # Count my contribution to the bid
    ones_wild = sim.ones_are_wild and not obs.get('maputa_active', False)
    my_contribution = sum(1 for d in my_hand if d == bid_face or (ones_wild and d == 1 and bid_face != 1))
    
    # Calculate needed dice from opponents
    needed_from_opponents = max(0, bid_qty - my_contribution)
    
    if needed_from_opponents <= 0:
        return 1.0  # We already have enough
    
    # Probability calculation for opponents' dice
    if bid_face == 1 or not ones_wild:
        # Probability is 1/6 per die
        prob_per_die = 1.0 / 6.0
    else:
        # Probability is 1/3 per die (target face + ones)
        prob_per_die = 1.0 / 3.0
    
    # Use binomial distribution to calculate probability
    return 1.0 - binom_cdf_ge_fast(opponent_dice, needed_from_opponents, prob_per_die)


def sample_dice_with_probabilities(rng, face_probs, num_dice):
    """Sample dice using specific face probabilities."""
    dice = []
    for _ in range(num_dice):
        # Use weighted random choice
        r = rng.random()
        cumsum = 0.0
        for face, prob in enumerate(face_probs, 1):
            cumsum += prob
            if r <= cumsum:
                dice.append(face)
                break
        else:
            # Fallback (should rarely happen)
            dice.append(6)
    return dice


def sample_bayesian_player_dice(agent, obs, player_idx, num_dice):
    """
    Sample dice for a specific player using Bayesian inference from betting history.
    
    Uses player's historical betting patterns to infer likely dice distributions:
    - Players who frequently bid on a face likely have more of that face
    - Adjust probabilities based on player trust and historical accuracy
    - Consider recent vs long-term patterns with recency weighting
    """
    # Check if betting history is available
    history = obs.get('betting_history')
    trust_params = obs.get('player_trust', [])
    
    if history is None or not trust_params:
        # Fall back to standard weighted sampling or uniform sampling
        if hasattr(agent, 'weighted_sampling') and agent.weighted_sampling:
            return sample_weighted_dice(agent, obs, num_dice)
        else:
            return [agent.rng.randint(1, 6) for _ in range(num_dice)]
    
    trust = trust_params[player_idx] if player_idx < len(trust_params) else 0.5
    
    # Get player's recent bidding patterns
    recent_bids = history.get_player_bids(player_idx)
    
    # Base uniform probabilities
    face_probs = [1.0/6.0] * 6
    
    # Adjust probabilities based on bidding history
    if recent_bids:
        # Consider last 5 bids or all available bids if fewer
        recent_limit = min(5, len(recent_bids))
        for bid_entry in recent_bids[-recent_limit:]:
            if bid_entry.action[0] == 'bid':
                bid_face = bid_entry.action[2]
                
                # Weight adjustment based on trust and recency
                recency_weight = calculate_recency_weight(bid_entry.round_num, history.current_round)
                trust_weight = trust  # Higher trust = more influence
                
                adjustment = 0.1 * trust_weight * recency_weight
                
                # Increase probability for bid face, decrease for others
                face_probs[bid_face - 1] += adjustment
                for i in range(6):
                    if i != bid_face - 1:
                        face_probs[i] -= adjustment / 5
    
    # Ensure all probabilities are non-negative
    face_probs = [max(0.01, p) for p in face_probs]
    
    # Normalize probabilities
    total_prob = sum(face_probs)
    face_probs = [p / total_prob for p in face_probs]
    
    # Sample dice using adjusted probabilities
    return sample_dice_with_probabilities(agent.rng, face_probs, num_dice)


def compute_enhanced_action_score(obs, action, statistical_prior):
    """Compute enhanced multi-criteria score for action pruning."""
    if action[0] != 'bid':
        return 1.0  # Non-bid actions get default high score
    
    qty, face = action[1], action[2]
    sim = obs['_simulator']
    current_bid = obs['current_bid']
    dice_counts = obs['dice_counts']
    my_hand = obs['my_hand']
    
    # Start with statistical prior (existing logic)
    score = statistical_prior
    
    # Opponent modeling heuristics
    total_dice = sum(dice_counts)
    my_dice_count = len(my_hand)
    opponent_dice = total_dice - my_dice_count
    
    # Factor 1: Bid aggressiveness relative to remaining dice
    if total_dice > 0:
        bid_ratio = qty / total_dice
        # Moderate bids (0.3-0.6 ratio) get slight boost, extreme bids get penalty
        if 0.3 <= bid_ratio <= 0.6:
            score *= 1.1  # Slight boost for reasonable bids
        elif bid_ratio > 0.8:
            score *= 0.8  # Penalty for very aggressive bids
    
    # Factor 2: Opponent dice distribution consideration
    if current_bid is not None:
        current_qty, current_face = current_bid
        # If opponents have significantly fewer dice, they may be more likely to call
        weak_opponents = sum(1 for dc in dice_counts if dc > 0 and dc <= 2)
        total_opponents = sum(1 for dc in dice_counts if dc > 0) - 1  # Exclude self
        
        if total_opponents > 0:
            weak_ratio = weak_opponents / total_opponents
            if weak_ratio > 0.5:  # Most opponents are weak
                # Be more conservative with large increases
                if qty > current_qty + 2:
                    score *= 0.9  # Slight penalty for large jumps when opponents are weak
    
    # Factor 3: Face transition considerations
    if current_bid is not None:
        current_qty, current_face = current_bid
        # Transitions involving ones have special strategic value
        if face == 1 and current_face != 1:
            # Transitioning to ones - often strategic
            score *= 1.05
        elif face != 1 and current_face == 1:
            # Transitioning from ones - requires careful calculation
            expected_ones_equivalent = current_qty * 2 + 1  # Standard conversion
            if qty >= expected_ones_equivalent:
                score *= 1.02  # Proper transition gets small boost
            else:
                score *= 0.95  # Improper transition gets penalty
    
    # Factor 4: Hand strength consideration
    my_face_count = sum(1 for d in my_hand if d == face)
    ones_count = sum(1 for d in my_hand if d == 1)
    
    # If we have good support for the bid face
    ones_wild = sim.ones_are_wild and not obs.get('maputa_active', False)
    total_support = my_face_count
    if ones_wild and face != 1:
        total_support += ones_count
    
    if total_support >= 2:  # Strong hand support
        score *= 1.08
    elif total_support == 0:  # No support at all
        score *= 0.92
    
    return max(0.0, score)  # Ensure non-negative score


def compute_control_variate_baseline(obs, action):
    """Compute baseline value for control variate variance reduction."""
    # Simple heuristic baseline based on bid plausibility and hand strength
    if action[0] != 'bid':
        # For call/exact actions, use simple probability based on current bid
        current_bid = obs['current_bid']
        if current_bid is None:
            return 0.5  # Neutral baseline for first move
        
        # Estimate call success probability using similar logic to baseline agent
        sim = obs['_simulator']
        my_hand = obs['my_hand']
        qty, face = current_bid
        k = sum(1 for d in my_hand if d == face)
        n_other = sum(obs['dice_counts']) - len(my_hand)
        
        if face != 1 and obs.get('maputa_active', False) is False and sim.ones_are_wild:
            p = 1 / 3
        else:
            p = 1 / 6
        
        need = max(0, qty - k)
        prob_true = binom_cdf_ge_fast(n_other, need, p) if n_other >= 0 else 0.0
        
        if action[0] == 'call':
            return 1.0 - prob_true  # Call succeeds when bid is false
        else:  # exact
            # Exact is harder to estimate, use conservative baseline
            return 0.3
    else:
        # For bid actions, use hand strength and bid reasonableness
        qty, face = action[1], action[2]
        my_hand = obs['my_hand']
        
        # Hand support for this bid
        my_face_count = sum(1 for d in my_hand if d == face)
        ones_count = sum(1 for d in my_hand if d == 1)
        
        sim = obs['_simulator']
        ones_wild = sim.ones_are_wild and not obs.get('maputa_active', False)
        total_support = my_face_count
        if ones_wild and face != 1:
            total_support += ones_count
        
        # Normalize support to hand size
        hand_strength = total_support / len(my_hand) if len(my_hand) > 0 else 0
        
        # Bid reasonableness (simple ratio to total dice)
        total_dice = sum(obs['dice_counts'])
        bid_ratio = qty / total_dice if total_dice > 0 else 0
        
        # Reasonable bids with good hand support get higher baseline
        if bid_ratio <= 0.6 and hand_strength >= 0.3:
            return 0.6
        elif bid_ratio <= 0.8 and hand_strength >= 0.2:
            return 0.4
        else:
            return 0.3


def apply_variance_reduction(simulation_results, obs, action):
    """Apply variance reduction techniques to simulation results."""
    if len(simulation_results) <= 1:
        # No variance reduction or insufficient data
        return sum(simulation_results) / len(simulation_results)
    
    # Control variate method
    baseline = compute_control_variate_baseline(obs, action)
    
    # Compute control variate correlation
    mean_result = sum(simulation_results) / len(simulation_results)
    
    # Simple control variate: adjust based on deviation from baseline
    # The control variate reduces variance by using the known baseline expectation
    deviations = [r - baseline for r in simulation_results]
    mean_deviation = sum(deviations) / len(deviations)
    
    # Reduce variance by adjusting for systematic deviation from baseline
    # This is a simplified version of the control variate method
    adjustment_factor = 0.3  # Conservative adjustment to avoid over-correction
    adjusted_mean = mean_result - (adjustment_factor * mean_deviation)
    
    # Ensure result stays within valid probability bounds
    return max(0.0, min(1.0, adjusted_mean))


def heuristic_win_prob(dice_counts, player_idx):
    """
    Simple heuristic value: given dice counts, approximate win probability.
    Logistic based on relative dice counts.
    """
    total = sum(dice_counts)
    if total <= 0:
        return 0.0
    my = dice_counts[player_idx]
    # baseline: probability proportional to dice count normalized
    # add small exponent to favor larger counts
    denom = sum((c ** 1.2) for c in dice_counts if c > 0)
    if denom <= 0:
        return 0.0
    return (my ** 1.2) / denom