# ISMCTS Implementation Specification

## Overview

This document specifies the complete implementation of Information Set Monte Carlo Tree Search (ISMCTS) for the Perudo game, replacing the current Monte Carlo agent with a proper ISMCTS algorithm. This specification follows Test-Driven Development (TDD) principles and provides a pure ISMCTS implementation without hybrid approaches.

## ISMCTS Algorithm Background

ISMCTS (Cowling et al., 2012) is designed for imperfect information games where players cannot see all game state information. The canonical algorithm consists of:

1. **Determinization**: Sample a "possible world" consistent with current information
2. **Selection**: Traverse the tree using UCT policy until reaching a leaf node
3. **Expansion**: Add a new child node for the selected action
4. **Simulation**: Play out the game from the new node to terminal state
5. **Backpropagation**: Update statistics for information set nodes in the path

## Architecture Design

### Core Components

#### 1. Information Set Node (`ISNode`)
```python
class ISNode:
    def __init__(self, info_key, parent=None):
        self.info_key = info_key          # Observable state hash
        self.parent = parent              # Parent ISNode
        self.children = {}                # action -> ISNode mapping
        self.visits = 0                   # Visit count
        self.value_sum = 0.0             # Cumulative value
        self.untried_actions = None       # Actions not yet expanded
        
    def ucb_score(self, c_param=1.414):
        """UCB1 score for action selection."""
        if self.visits == 0:
            return float('inf')
        exploitation = self.value_sum / self.visits
        exploration = c_param * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration
```

#### 2. ISMCTS Agent (`ISMCTSAgent`)
```python
class ISMCTSAgent:
    def __init__(self, n_iterations=500, c_param=1.414, **mc_kwargs):
        # Inherit all existing MC capabilities
        super().__init__(**mc_kwargs)
        self.n_iterations = n_iterations
        self.c_param = c_param
        self.tree = {}  # info_key -> ISNode
        self.tree_memory_limit = 1000000  # Maximum nodes in tree
```

### Information State Representation

#### Hash Function Design
```python
def hash_info(self, obs):
    """Create hashable key for information set."""
    return (
        obs['player_idx'],
        tuple(obs['dice_counts']),
        obs['current_bid'],
        obs.get('maputa_active', False),
        obs.get('maputa_restrict_face', None),
        tuple(obs['my_hand'])  # Include own dice for complete info set
    )
```

The information set includes:
- Current player position
- Public dice counts for all players
- Current bid on the table
- Maputa status and restrictions
- Player's own dice (known information)

## Implementation Phases

### Phase 1: Core ISMCTS Infrastructure

**Objective**: Implement basic ISMCTS node structure and tree operations.

**Components**:
1. `ISNode` class with UCB scoring
2. Information set hashing function
3. Basic tree memory management
4. Node creation and relationship handling

**Tests Required**:
- Node creation and parent-child relationships
- UCB score calculations
- Information set hashing consistency
- Tree memory management

### Phase 2: ISMCTS Algorithm Implementation

**Objective**: Implement the core ISMCTS search algorithm.

**Components**:
1. Selection phase with UCT policy
2. Expansion phase with untried actions
3. Simulation using existing rollout policies
4. Backpropagation to information set nodes
5. Action selection from root node

**Algorithm Flow**:
```python
def _ismcts_iteration(self, root, obs):
    # 1. Determinization
    hands = self.sample_determinization(obs)
    
    # 2. Selection & Expansion
    path = []
    current_node = root
    current_obs = obs.copy()
    
    while True:
        # Initialize untried actions on first visit
        if current_node.untried_actions is None:
            current_node.untried_actions = list(self._get_legal_actions(current_obs))
        
        # Expansion: try untried action first
        if current_node.untried_actions:
            action = current_node.untried_actions.pop()
            new_obs = self._apply_action(current_obs, action, hands)
            
            if new_obs is None:  # Terminal state
                reward = self._evaluate_terminal(current_obs, action, hands)
                break
                
            new_key = self.hash_info(new_obs)
            child_node = current_node.add_child(action, new_key)
            path.append((current_node, action))
            
            # Simulation from new node
            reward = self.simulate_from_determinization(hands, new_obs, None)
            break
            
        # Selection: choose best child via UCB
        if not current_node.children:
            # Terminal node
            reward = self._evaluate_terminal(current_obs, None, hands)
            break
            
        action, child_node = max(
            current_node.children.items(),
            key=lambda x: x[1].ucb_score(self.c_param)
        )
        
        path.append((current_node, action))
        current_obs = self._apply_action(current_obs, action, hands)
        
        if current_obs is None:  # Terminal
            reward = self._evaluate_terminal(current_obs, action, hands)
            break
            
        current_node = child_node
    
    # 3. Backpropagation
    self._backpropagate(path, reward)
```

### Phase 3: Integration with Existing Features

**Objective**: Preserve and integrate existing Monte Carlo enhancements.

**Preserved Features**:
- Advanced determinization sampling (Bayesian, weighted)
- Player trust modeling and betting history
- Variance reduction techniques
- Enhanced action pruning
- Parallel processing capabilities

**Integration Strategy**:
- Reuse existing `sample_determinization()` method
- Maintain `simulate_from_determinization()` for rollouts
- Preserve configuration system and parameters
- Keep all existing utility functions

### Phase 4: Tree Management and Optimization

**Objective**: Implement efficient tree management for long-term play.

**Components**:
1. Tree pruning strategies
2. Memory management
3. Progressive widening for large action spaces
4. Tree persistence between turns

**Tree Management**:
```python
def _manage_tree_memory(self):
    """Prune tree to stay within memory limits."""
    if len(self.tree) > self.tree_memory_limit:
        # Remove least visited nodes
        sorted_nodes = sorted(self.tree.items(), key=lambda x: x[1].visits)
        nodes_to_remove = len(self.tree) - self.tree_memory_limit + 100
        
        for info_key, node in sorted_nodes[:nodes_to_remove]:
            del self.tree[info_key]
```

### Phase 5: Performance Optimization

**Objective**: Optimize ISMCTS for real-time play performance.

**Optimizations**:
1. Efficient node storage and lookup
2. Fast information set hashing
3. Optimized UCB calculations
4. Parallel ISMCTS with multiple trees
5. Progressive widening implementation

## Configuration Integration

### Updated Configuration Schema
```json
{
    "agent_type": "ismcts",
    "ismcts_iterations": 500,
    "ismcts_c_param": 1.414,
    "tree_memory_limit": 1000000,
    "progressive_widening_enabled": false,
    "progressive_widening_threshold": 10,
    
    // Preserve all existing MC parameters
    "weighted_sampling": true,
    "bayesian_sampling": true,
    "betting_history_enabled": true,
    "player_trust_enabled": true,
    "variance_reduction": true,
    "enhanced_pruning": true,
    "enable_parallel": true
}
```

### Parameter Guidelines
- `ismcts_iterations`: 100-1000 depending on time constraints
- `ismcts_c_param`: 1.414 (√2) is standard, may tune between 0.5-2.0
- `tree_memory_limit`: Based on available RAM (1M nodes ≈ 100MB)

## Testing Strategy

### Unit Tests
1. **Node Operations**:
   - Node creation and initialization
   - UCB score calculations
   - Parent-child relationships
   - Action storage and retrieval

2. **Tree Operations**:
   - Information set hashing
   - Tree traversal
   - Node expansion
   - Memory management

3. **Algorithm Components**:
   - Selection phase logic
   - Expansion phase logic
   - Backpropagation correctness
   - Terminal state handling

### Integration Tests
1. **ISMCTS vs Existing Agents**:
   - Tournament performance comparison
   - Decision quality assessment
   - Computational efficiency metrics

2. **Feature Preservation**:
   - Verify all MC enhancements still work
   - Configuration system integration
   - Parallel processing compatibility

### Performance Tests
1. **Scalability**:
   - Tree growth patterns
   - Memory usage over time
   - Iteration speed benchmarks

2. **Game Performance**:
   - Decision time measurements
   - Win rate improvements
   - Strategic behavior analysis

## Implementation Steps

### Week 1: Core Infrastructure
1. Implement `ISNode` class with comprehensive tests
2. Create information set hashing function
3. Add basic tree management
4. Set up test framework for ISMCTS components

**Deliverables**:
- `ISNode` class implementation
- Unit tests for node operations
- Information set hashing with tests
- Basic tree memory management

### Week 2: Algorithm Implementation
1. Implement core ISMCTS iteration logic
2. Add selection and expansion phases
3. Implement backpropagation
4. Create action selection from root

**Deliverables**:
- Complete ISMCTS iteration algorithm
- Selection and expansion logic
- Backpropagation implementation
- Basic action selection

### Week 3: Integration and Configuration
1. Create `ISMCTSAgent` class inheriting from `MonteCarloAgent`
2. Integrate with existing configuration system
3. Preserve all existing MC enhancements
4. Add comprehensive integration tests

**Deliverables**:
- Complete `ISMCTSAgent` implementation
- Configuration integration
- Feature preservation verification
- Integration test suite

### Week 4: Optimization and Validation
1. Implement tree management optimizations
2. Add performance monitoring
3. Conduct tournament comparisons
4. Fine-tune parameters

**Deliverables**:
- Optimized tree management
- Performance benchmarking
- Tournament evaluation results
- Parameter tuning recommendations

## Expected Benefits

### Performance Improvements
1. **Search Efficiency**: Reuse computation across similar game states
2. **Memory of Experience**: Tree maintains learned action values
3. **Progressive Learning**: Decision quality improves over multiple turns
4. **Better Exploration**: UCB balances exploration vs exploitation

### Strategic Advantages
1. **Structured Search**: Systematic exploration of action space
2. **Information Set Awareness**: Proper handling of hidden information
3. **Long-term Planning**: Tree structure supports multi-step reasoning
4. **Adaptive Behavior**: Agent learns opponent patterns over time

## Risk Mitigation

### Potential Issues
1. **Memory Growth**: Tree may consume excessive memory in long games
2. **Cold Start**: Initial decisions before tree builds may be poor
3. **Computational Overhead**: UCB calculations and tree operations add cost
4. **Implementation Complexity**: More complex than simple MC sampling

### Mitigation Strategies
1. **Memory Management**: Implement aggressive tree pruning
2. **Fallback Mechanisms**: Use existing MC logic for initial decisions
3. **Performance Monitoring**: Track computation time and optimize bottlenecks
4. **Incremental Development**: Implement and test components gradually

## Success Metrics

### Quantitative Measures
1. **Win Rate Improvement**: >5% increase in tournament performance
2. **Decision Time**: Maintain <2s per decision in web play
3. **Memory Usage**: <200MB tree size in typical games
4. **Computational Efficiency**: >100 iterations per second

### Qualitative Measures
1. **Strategic Behavior**: More consistent and logical decision patterns
2. **Bluffing Detection**: Better recognition of opponent strategies
3. **End-game Play**: Improved performance in final rounds
4. **Code Quality**: Maintainable, well-tested implementation

## Conclusion

This specification provides a complete roadmap for implementing ISMCTS in the Perudo project. The implementation preserves all existing Monte Carlo enhancements while adding the structured search and memory benefits of ISMCTS. The phased approach ensures manageable development with comprehensive testing at each stage.

The pure ISMCTS implementation will provide superior decision-making capabilities through structured search and information set awareness, making it a significant upgrade over the current Monte Carlo sampling approach.