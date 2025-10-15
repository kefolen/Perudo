"""
Monte Carlo Agent Utilities

This module contains helper functions for the MonteCarloAgent including:
- Weighted determinization sampling based on bidding history
- Enhanced action scoring for advanced pruning
- Control variate baseline calculations for variance reduction
- Heuristic win probability estimation
"""

import random
from sim.perudo import binom_cdf_ge_fast


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