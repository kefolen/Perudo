"""
Test scenarios for edge cases and special situations in Perudo.

This module provides test scenarios that cover edge cases, boundary conditions,
and special rule interactions for comprehensive testing coverage.
"""

from sim.perudo import Action


class EdgeCases:
    """Edge case scenarios for boundary testing"""
    
    @staticmethod
    def single_player_remaining():
        """Edge case: Only one player left (game should end)"""
        return {
            'dice_counts': [3, 0, 0],  # Only player 0 has dice
            'player_idx': 0,
            'my_hand': [1, 2, 3],
            'current_bid': None,
            'maputa_restrict_face': None,
            'round_num': 15
        }
    
    @staticmethod
    def all_players_one_die():
        """Edge case: All players down to single die"""
        return {
            'dice_counts': [1, 1, 1, 1],  # 4 players, all with 1 die
            'player_idx': 2,
            'my_hand': [4],
            'current_bid': (1, 3),
            'maputa_restrict_face': 3,  # Maputa active
            'round_num': 12
        }
    
    @staticmethod
    def maximum_players_start():
        """Edge case: Maximum number of players at game start"""
        return {
            'dice_counts': [5, 5, 5, 5, 5, 5],  # 6 players
            'player_idx': 3,
            'my_hand': [1, 1, 2, 4, 6],
            'current_bid': (4, 2),
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def minimum_players():
        """Edge case: Minimum number of players (2)"""
        return {
            'dice_counts': [3, 2],  # Only 2 players
            'player_idx': 1,
            'my_hand': [5, 6],
            'current_bid': (2, 1),
            'maputa_restrict_face': None,
            'round_num': 5
        }
    
    @staticmethod
    def zero_dice_edge():
        """Edge case: Player with zero dice (eliminated)"""
        return {
            'dice_counts': [2, 0, 3],  # Player 1 eliminated
            'player_idx': 2,
            'my_hand': [1, 2, 3],
            'current_bid': (1, 4),
            'maputa_restrict_face': None,
            'round_num': 8
        }


class BoundaryConditions:
    """Boundary condition scenarios"""
    
    @staticmethod
    def maximum_possible_bid():
        """Boundary: Maximum theoretically possible bid"""
        return {
            'dice_counts': [5, 5, 5, 5, 5, 5],  # 30 dice total
            'player_idx': 0,
            'my_hand': [6, 6, 6, 6, 6],
            'current_bid': (29, 6),  # Near maximum
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def minimum_valid_bid():
        """Boundary: Minimum valid bid (1, 2)"""
        return {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,  # No current bid
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def face_value_boundaries():
        """Boundary: Testing face values 1 and 6"""
        return [
            {
                'dice_counts': [3, 3, 3],
                'player_idx': 0,
                'my_hand': [1, 1, 1],  # All ones
                'current_bid': (2, 1),
                'maputa_restrict_face': None,
                'round_num': 3
            },
            {
                'dice_counts': [3, 3, 3],
                'player_idx': 1,
                'my_hand': [6, 6, 6],  # All sixes
                'current_bid': (2, 6),
                'maputa_restrict_face': None,
                'round_num': 3
            }
        ]
    
    @staticmethod
    def quantity_boundaries():
        """Boundary: Testing quantity limits"""
        return {
            'dice_counts': [1, 1, 1],  # Only 3 dice total
            'player_idx': 0,
            'my_hand': [2],
            'current_bid': (3, 2),  # Bid equals total dice
            'maputa_restrict_face': None,
            'round_num': 10
        }


class SpecialRuleInteractions:
    """Scenarios testing special rule interactions"""
    
    @staticmethod
    def maputa_with_ones_wild():
        """Maputa restriction with ones being wild"""
        return {
            'dice_counts': [2, 1, 3],
            'player_idx': 1,  # Player with 1 die
            'my_hand': [1],   # Has a one
            'current_bid': (2, 3),  # Bid on 3s
            'maputa_restrict_face': 3,  # Maputa restricts to 3s
            'round_num': 7,
            'ones_are_wild': True
        }
    
    @staticmethod
    def maputa_without_ones_wild():
        """Maputa restriction without ones being wild"""
        return {
            'dice_counts': [2, 1, 3],
            'player_idx': 1,
            'my_hand': [4],
            'current_bid': (1, 2),
            'maputa_restrict_face': 2,
            'round_num': 6,
            'ones_are_wild': False
        }
    
    @staticmethod
    def exact_call_scenarios():
        """Various exact call scenarios"""
        return [
            {
                'name': 'exact_call_likely_true',
                'dice_counts': [2, 2, 2],
                'player_idx': 1,
                'my_hand': [4, 4],
                'current_bid': (3, 4),  # Likely exactly 3 fours
                'maputa_restrict_face': None,
                'round_num': 5
            },
            {
                'name': 'exact_call_risky',
                'dice_counts': [3, 3, 2],
                'player_idx': 2,
                'my_hand': [2, 5],
                'current_bid': (4, 2),  # Risky exact call
                'maputa_restrict_face': None,
                'round_num': 4
            }
        ]
    
    @staticmethod
    def ones_wild_transitions():
        """Scenarios involving transitions between ones and regular faces"""
        return [
            {
                'name': 'regular_to_ones_minimum',
                'dice_counts': [4, 4, 3],
                'player_idx': 0,
                'my_hand': [1, 2, 3, 4],
                'current_bid': (4, 6),  # High regular bid
                'maputa_restrict_face': None,
                'round_num': 2
            },
            {
                'name': 'ones_to_regular_calculation',
                'dice_counts': [3, 3, 3],
                'player_idx': 1,
                'my_hand': [1, 1, 5],
                'current_bid': (3, 1),  # Ones bid
                'maputa_restrict_face': None,
                'round_num': 3
            }
        ]


class ErrorConditions:
    """Scenarios that should trigger error handling"""
    
    @staticmethod
    def invalid_game_states():
        """Invalid game states that should be handled gracefully"""
        return [
            {
                'name': 'negative_dice_count',
                'dice_counts': [-1, 3, 2],  # Invalid negative count
                'player_idx': 0,
                'my_hand': [1, 2, 3],
                'current_bid': (2, 3),
                'maputa_restrict_face': None,
                'round_num': 1
            },
            {
                'name': 'mismatched_hand_size',
                'dice_counts': [2, 3, 2],
                'player_idx': 0,
                'my_hand': [1, 2, 3, 4, 5],  # Hand size doesn't match dice_count
                'current_bid': (1, 2),
                'maputa_restrict_face': None,
                'round_num': 1
            },
            {
                'name': 'invalid_player_index',
                'dice_counts': [3, 3, 3],
                'player_idx': 5,  # Index out of range
                'my_hand': [1, 2, 3],
                'current_bid': (2, 4),
                'maputa_restrict_face': None,
                'round_num': 1
            }
        ]
    
    @staticmethod
    def invalid_bids():
        """Invalid bid scenarios"""
        return [
            {
                'name': 'zero_quantity',
                'dice_counts': [3, 3, 3],
                'player_idx': 0,
                'my_hand': [1, 2, 3],
                'current_bid': (0, 3),  # Zero quantity
                'maputa_restrict_face': None,
                'round_num': 1
            },
            {
                'name': 'invalid_face',
                'dice_counts': [3, 3, 3],
                'player_idx': 1,
                'my_hand': [2, 4, 6],
                'current_bid': (2, 7),  # Face > 6
                'maputa_restrict_face': None,
                'round_num': 1
            },
            {
                'name': 'impossible_quantity',
                'dice_counts': [2, 1, 1],
                'player_idx': 2,
                'my_hand': [3],
                'current_bid': (50, 2),  # Impossible quantity
                'maputa_restrict_face': None,
                'round_num': 8
            }
        ]


class PerformanceScenarios:
    """Scenarios designed to test performance edge cases"""
    
    @staticmethod
    def large_game_state():
        """Large game with maximum players and dice"""
        return {
            'dice_counts': [5] * 6,  # 6 players, 5 dice each
            'player_idx': 3,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': (8, 3),
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def complex_bid_calculation():
        """Scenario requiring complex probability calculations"""
        return {
            'dice_counts': [4, 3, 4, 2, 3],  # Uneven distribution
            'player_idx': 2,
            'my_hand': [1, 1, 2, 6],
            'current_bid': (7, 2),  # Complex probability scenario
            'maputa_restrict_face': None,
            'round_num': 4
        }
    
    @staticmethod
    def rapid_game_progression():
        """Scenario simulating rapid game state changes"""
        states = []
        dice_counts = [5, 5, 5]
        
        for round_num in range(1, 10):
            # Simulate dice loss over rounds
            if round_num > 3:
                dice_counts = [max(1, d-1) for d in dice_counts]
            
            states.append({
                'dice_counts': dice_counts.copy(),
                'player_idx': round_num % 3,
                'my_hand': [1, 2] if dice_counts[round_num % 3] == 2 else [3],
                'current_bid': (round_num, min(6, round_num + 1)),
                'maputa_restrict_face': None if dice_counts[round_num % 3] > 1 else round_num % 6 + 1,
                'round_num': round_num
            })
        
        return states


class RegressionScenarios:
    """Scenarios for regression testing specific fixes"""
    
    @staticmethod
    def mc_agent_parameter_change():
        """Scenarios specifically for testing MC_N parameter reduction"""
        return [
            {
                'name': 'mc_decision_quality_test',
                'dice_counts': [3, 3, 3],
                'player_idx': 1,
                'my_hand': [2, 3, 4],
                'current_bid': (4, 2),
                'maputa_restrict_face': None,
                'round_num': 3,
                'expected_action_type': 'call'  # Should likely call this bid
            },
            {
                'name': 'mc_performance_test',
                'dice_counts': [4, 4, 4],
                'player_idx': 0,
                'my_hand': [1, 1, 3, 5],
                'current_bid': (3, 3),
                'maputa_restrict_face': None,
                'round_num': 2,
                'time_limit_seconds': 5  # Should complete within time limit
            }
        ]
    
    @staticmethod
    def backward_compatibility():
        """Scenarios ensuring backward compatibility"""
        return {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,
            'maputa_restrict_face': None,
            'round_num': 1,
            'legacy_format': True  # Flag for legacy compatibility testing
        }