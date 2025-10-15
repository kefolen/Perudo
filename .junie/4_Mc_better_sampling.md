# Monte Carlo Agent Better Sampling Enhancement Specification

This document outlines the enhancement plan for the Monte Carlo Agent to implement history-aware betting analysis and improved opponent modeling through Bayesian probability sampling and dynamic player trust parameters.

## Overview

The goal is to enhance the existing Monte Carlo Agent with sophisticated betting history tracking and analysis capabilities that will significantly improve decision-making quality through better opponent modeling and more accurate hand sampling.

### Current State Analysis

The existing MC Agent has sophisticated features but lacks historical context:
- **Weighted Determinization**: Currently only considers immediate current_bid for sampling weights
- **Enhanced Action Scoring**: Uses basic opponent modeling but no historical patterns
- **No Betting History**: Each decision is made in isolation without learning from past rounds
- **Static Trust**: All players are treated equally regardless of past bidding accuracy
- **Limited Plausibility**: Bid evaluation only considers current state, not collective patterns

### Enhancement Objectives

This specification introduces three major enhancements:

1. **Betting History Tracking**: Comprehensive tracking of all player bids throughout the game
2. **Collective Plausibility Analysis**: Better bid evaluation using all players' historical patterns  
3. **Bayesian Player Modeling**: Individual player dice sampling based on betting history correlation
4. **Dynamic Player Trust**: Adaptive trust parameters that adjust based on bidding accuracy over time

## Technical Approach

### 1. Betting History Tracking System

#### Data Structures

```python
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
        
    def add_entry(self, entry):
        """Add a new betting entry to the history."""
        
    def get_player_bids(self, player_idx, face=None):
        """Get all bids made by a specific player, optionally filtered by face."""
        
    def get_face_popularity(self, face, recent_rounds=None):
        """Calculate how frequently a face has been bid on by all players."""
        
    def analyze_player_accuracy(self, player_idx):
        """Analyze how accurate a player's bids have been historically."""
```

#### Integration with Game Loop

The betting history will be maintained in the observation structure:

```python
# Enhanced observation structure
obs = {
    'player_idx': current_player,
    'my_hand': list(hands[current_player]),
    'dice_counts': list(state['dice_counts']),
    'current_bid': current_bid,
    'betting_history': game_betting_history,  # New field
    'player_trust': player_trust_params,       # New field
    'maputa_active': maputa_active,
    'maputa_restrict_face': maputa_restrict_face,
    '_simulator': self,
}
```

### 2. Collective Plausibility Analysis

#### Enhanced Plausibility Calculation

```python
def compute_collective_plausibility(obs, bid_qty, bid_face):
    """
    Calculate bid plausibility based on collective betting patterns.
    
    Factors considered:
    1. How often this face has been bid on by all players
    2. Historical accuracy of bids on this face
    3. Current game state vs historical patterns
    4. Player trust levels for those who have bid this face
    """
    history = obs['betting_history']
    trust_params = obs['player_trust']
    
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
```

### 3. Bayesian Player Modeling

#### Individual Player Dice Sampling

```python
def sample_bayesian_player_dice(agent, obs, player_idx, num_dice):
    """
    Sample dice for a specific player using Bayesian inference from betting history.
    
    Uses player's historical betting patterns to infer likely dice distributions:
    - Players who frequently bid on a face likely have more of that face
    - Adjust probabilities based on player trust and historical accuracy
    - Consider recent vs long-term patterns with recency weighting
    """
    history = obs['betting_history']
    trust = obs['player_trust'][player_idx]
    
    # Get player's recent bidding patterns
    recent_bids = history.get_player_bids(player_idx)
    
    # Base uniform probabilities
    face_probs = [1/6] * 6
    
    # Adjust probabilities based on bidding history
    for bid_entry in recent_bids[-5:]:  # Consider last 5 bids
        if bid_entry.action[0] == 'bid':
            bid_face = bid_entry.action[2]
            
            # Weight adjustment based on trust and recency
            recency_weight = calculate_recency_weight(bid_entry.round_num, obs['current_round'])
            trust_weight = trust  # Higher trust = more influence
            
            adjustment = 0.1 * trust_weight * recency_weight
            
            # Increase probability for bid face, decrease for others
            face_probs[bid_face - 1] += adjustment
            for i in range(6):
                if i != bid_face - 1:
                    face_probs[i] -= adjustment / 5
    
    # Normalize probabilities
    total_prob = sum(face_probs)
    face_probs = [p / total_prob for p in face_probs]
    
    # Sample dice using adjusted probabilities
    return sample_dice_with_probabilities(agent.rng, face_probs, num_dice)
```

### 4. Dynamic Player Trust System

#### Trust Parameter Management

```python
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
                actual_face_count = sum(1 for d in bid_entry.actual_hand if d == bid_face)
                expected_proportion = actual_face_count / len(bid_entry.actual_hand)
                
                # Higher accuracy for players whose bids correlate with their actual dice
                base_accuracy = min(1.0, expected_proportion * 2)  # Scale to reasonable range
                total_accuracy += base_accuracy
        
        return total_accuracy / len(player_bids)
```

## Implementation Strategy

### Phase 5: Betting History and Player Trust (Week 1-2)

**Objective**: Implement comprehensive betting history tracking and dynamic player trust system

**Key Components**:
- Extend simulator to track betting history in observation structure
- Implement `BettingHistoryEntry` and `GameBettingHistory` classes
- Add `PlayerTrustManager` for dynamic trust parameter updates
- Enhance existing observation structure with new fields
- Maintain full backward compatibility

**Configuration Parameters**:
```python
MonteCarloAgent(
    # Existing parameters preserved
    n=200, chunk_size=8, max_rounds=6,
    weighted_sampling=False, enable_parallel=False,
    enhanced_pruning=False, variance_reduction=False,
    
    # New Phase 5 parameters
    betting_history_enabled=False,    # Enable betting history tracking
    player_trust_enabled=False,       # Enable dynamic trust parameters
    trust_learning_rate=0.1,          # How quickly trust adapts
    history_memory_rounds=10,         # How many rounds to remember
    bayesian_sampling=False,          # Enable Bayesian player dice sampling
)
```

### Phase 6: Enhanced Sampling and Plausibility (Week 2-3)

**Objective**: Implement Bayesian player modeling and collective plausibility analysis

**Key Enhancements**:
- Replace `sample_weighted_dice` with `sample_bayesian_weighted_dice`
- Implement `compute_collective_plausibility` function
- Enhance `compute_enhanced_action_score` with historical data
- Add face popularity and accuracy tracking
- Integrate trust parameters into all sampling decisions

**New Functions**:
```python
# Enhanced sampling with Bayesian inference
def sample_bayesian_weighted_dice(agent, obs, player_idx, num_dice)

# Collective plausibility with historical context  
def compute_collective_plausibility(obs, bid_qty, bid_face)

# Trust-weighted accuracy calculations
def calculate_trust_weighted_accuracy(history, trust_params, face)
```

## Expected Outcomes

### Performance Improvements

- **Enhanced Decision Quality**: 15-25% improvement in tournament win rates through better opponent modeling
- **Adaptive Learning**: Agents become more effective against the same opponents over multiple games
- **Strategic Depth**: More sophisticated bluffing and calling decisions based on historical patterns
- **Player-Specific Tactics**: Customized strategies for different opponent types

### Strategic Benefits

- **Pattern Recognition**: Ability to detect and exploit opponent betting patterns
- **Trust-Based Decisions**: More accurate risk assessment based on opponent reliability
- **Collective Intelligence**: Better understanding of table dynamics and face popularity
- **Adaptive Bluffing**: Dynamic bluffing strategies based on perceived player trust levels

## Testing Strategy

Following TDD principles, comprehensive tests must be implemented:

### Unit Tests (15-20 tests)
- `TestBettingHistoryTracking`: History data structure functionality
- `TestPlayerTrustManager`: Trust parameter updates and accuracy calculations  
- `TestBayesianSampling`: Bayesian dice sampling with historical data
- `TestCollectivePlausibility`: Plausibility calculations with betting patterns

### Integration Tests (8-10 tests)
- `TestHistoryIntegration`: Full game integration with betting history
- `TestTrustUpdates`: End-to-end trust parameter evolution
- `TestBackwardCompatibility`: All existing functionality preserved

### Performance Tests (5-7 tests) 
- `TestHistoryPerformanceImpact`: Ensure acceptable computational overhead
- `TestMemoryUsage`: Validate betting history memory consumption
- `TestTournamentImprovement`: Measure win rate improvements with new features

## Validation Approach

### Effectiveness Metrics
- **Tournament Win Rate**: Compare agents with/without betting history features
- **Adaptation Speed**: Measure how quickly agents adapt to opponent patterns
- **Memory Efficiency**: Validate reasonable memory usage for history tracking
- **Decision Quality**: Analyze correlation between trust parameters and actual opponent behavior

### Regression Testing
- **Backward Compatibility**: All existing tests continue to pass
- **Parameter Preservation**: Default behavior identical to current implementation
- **Performance Baseline**: No degradation when new features are disabled

## Future Extensions

The enhanced betting history system enables advanced AI techniques:

### Advanced Learning
- **Neural Network Integration**: Use betting history as features for deep learning models
- **Opponent Type Classification**: Automatically categorize opponents based on betting patterns
- **Meta-Game Learning**: Adapt strategies based on population-level betting trends

### Advanced Inference
- **Hidden Information Inference**: Infer opponent hands from betting sequences
- **Bluff Detection**: Identify statistical patterns indicating deceptive bidding
- **Coalition Detection**: Detect potential cooperation between opponents

## Configuration and Backward Compatibility

All new features are opt-in with sensible defaults:

```python
# Default configuration (identical to current behavior)
agent = MonteCarloAgent()  # All new features disabled

# Incremental adoption
agent = MonteCarloAgent(betting_history_enabled=True)  # Just history tracking
agent = MonteCarloAgent(player_trust_enabled=True)     # Add trust parameters
agent = MonteCarloAgent(bayesian_sampling=True)        # Full Bayesian sampling

# Complete enhanced configuration
agent = MonteCarloAgent(
    betting_history_enabled=True,
    player_trust_enabled=True, 
    bayesian_sampling=True,
    trust_learning_rate=0.15,
    history_memory_rounds=8
)
```

The specification ensures that existing code continues to work unchanged while providing powerful new capabilities for advanced strategic play.