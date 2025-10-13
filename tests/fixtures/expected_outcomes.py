"""
Expected outcomes for various game scenarios in Perudo.

This module provides known correct results for game scenarios to validate
the correctness of game mechanics, probability calculations, and agent decisions.
"""

from sim.perudo import Action


class ExpectedLegalActions:
    """Expected legal actions for various game states"""

    @staticmethod
    def early_game_no_bid():
        """Expected legal actions when no bid exists (early game)"""
        # Should be able to bid any quantity 1-15 with any face 2-6, or quantity 1-7 with face 1
        expected_bids = []

        # Regular faces (2-6): can bid 1-15 of each
        for face in range(2, 7):
            for qty in range(1, 16):  # 15 dice total (3 players * 5 dice)
                expected_bids.append(('bid', qty, face))

        # Ones: can bid 1-7 (half of total dice, rounded down)
        for qty in range(1, 8):
            expected_bids.append(('bid', qty, 1))

        return expected_bids

    @staticmethod
    def mid_game_with_bid():
        """Expected legal actions after bid (2, 3) with 8 total dice"""
        # Current bid: (2, 3)
        # Total dice: 8 (3+2+3)
        expected_actions = []

        # Can increase quantity on same face
        for qty in range(3, 9):  # 3-8 threes
            expected_actions.append(('bid', qty, 3))

        # Can bid same quantity on higher faces
        for face in range(4, 7):  # faces 4, 5, 6
            expected_actions.append(('bid', 2, face))

        # Can switch to ones (half the quantity, rounded down)
        expected_actions.append(('bid', 1, 1))

        # Can call or exact
        expected_actions.append(('call',))
        expected_actions.append(('exact',))

        return expected_actions

    @staticmethod
    def maputa_restricted():
        """Expected legal actions with maputa restriction on face 3"""
        # Player with 1 die, maputa restricts to face 3
        # Current bid: (1, 3)
        expected_actions = [
            ('bid', 2, 3),  # Can only increase quantity on restricted face
            ('bid', 3, 3),
            ('bid', 4, 3),  # Up to total dice count
            ('call',),
            ('exact',)
        ]
        return expected_actions


class ExpectedProbabilities:
    """Expected probability calculations for various scenarios"""

    @staticmethod
    def simple_probability_scenarios():
        """Known probability calculations for validation"""
        return [
            {
                'scenario': 'bid_2_threes_with_8_dice',
                'total_dice': 8,
                'my_hand': [1, 2, 4, 5],  # No threes, but has a one (wild)
                'bid': (2, 3),
                'ones_are_wild': True,
                'expected_probability_range': (0.7, 0.9),  # Should be high probability
                'explanation': 'Need 1 more three from 4 other dice, p=1/3 per die'
            },
            {
                'scenario': 'bid_4_sixes_with_6_dice',
                'total_dice': 6,
                'my_hand': [6, 6],  # Have 2 sixes
                'bid': (4, 6),
                'ones_are_wild': False,
                'expected_probability_range': (0.1, 0.3),  # Should be low probability
                'explanation': 'Need 2 more sixes from 4 dice, p=1/6 per die'
            },
            {
                'scenario': 'bid_ones_scenario',
                'total_dice': 9,
                'my_hand': [1, 1, 3],  # Have 2 ones
                'bid': (3, 1),
                'ones_are_wild': True,
                'expected_probability_range': (0.6, 0.8),  # Moderate-high probability
                'explanation': 'Need 1 more one from 6 dice, p=1/6 per die'
            }
        ]

    @staticmethod
    def edge_case_probabilities():
        """Edge case probability calculations"""
        return [
            {
                'scenario': 'impossible_bid',
                'total_dice': 4,
                'my_hand': [2, 3],
                'bid': (10, 4),  # Impossible with only 4 dice
                'ones_are_wild': True,
                'expected_probability': 0.0,
                'explanation': 'Impossible bid should have 0 probability'
            },
            {
                'scenario': 'certain_bid',
                'total_dice': 6,
                'my_hand': [2, 2, 2],  # Have 3 twos
                'bid': (1, 2),  # Bid only 1 two
                'ones_are_wild': True,
                'expected_probability': 1.0,
                'explanation': 'Already have more than bid amount'
            }
        ]


class ExpectedGameOutcomes:
    """Expected outcomes for complete game scenarios"""

    @staticmethod
    def simple_call_outcomes():
        """Expected outcomes when calls are made"""
        return [
            {
                'scenario': 'successful_call',
                'all_hands': [[1, 2, 3], [4, 5, 6], [1, 2, 4]],  # 9 dice total
                'bid': (4, 2),  # Bid 4 twos
                'ones_are_wild': True,
                'actual_count': 3,  # Only 3 twos (including ones as wild)
                'expected_outcome': 'call_succeeds',
                'explanation': 'Bid is false, caller wins'
            },
            {
                'scenario': 'failed_call',
                'all_hands': [[1, 1, 2], [2, 2, 3], [1, 2, 4]],  # 9 dice total
                'bid': (4, 2),  # Bid 4 twos
                'ones_are_wild': True,
                'actual_count': 5,  # 5 twos (3 ones + 2 twos = 5)
                'expected_outcome': 'call_fails',
                'explanation': 'Bid is true, caller loses'
            }
        ]

    @staticmethod
    def exact_call_outcomes():
        """Expected outcomes for exact calls"""
        return [
            {
                'scenario': 'successful_exact',
                'all_hands': [[3, 3, 4], [3, 5, 6], [1, 2, 4]],  # 9 dice total
                'bid': (3, 3),  # Bid exactly 3 threes
                'ones_are_wild': True,
                'actual_count': 3,  # Exactly 3 threes (2 threes + 1 one)
                'expected_outcome': 'exact_succeeds',
                'explanation': 'Exact count matches bid'
            },
            {
                'scenario': 'failed_exact_too_few',
                'all_hands': [[3, 4, 5], [2, 5, 6], [1, 2, 4]],  # 9 dice total
                'bid': (3, 3),  # Bid exactly 3 threes
                'ones_are_wild': True,
                'actual_count': 2,  # Only 2 threes (1 three + 1 one)
                'expected_outcome': 'exact_fails',
                'explanation': 'Count is less than bid'
            },
            {
                'scenario': 'failed_exact_too_many',
                'all_hands': [[3, 3, 3], [3, 5, 6], [1, 2, 4]],  # 9 dice total
                'bid': (3, 3),  # Bid exactly 3 threes
                'ones_are_wild': True,
                'actual_count': 4,  # 4 threes (3 threes + 1 one)
                'expected_outcome': 'exact_fails',
                'explanation': 'Count is more than bid'
            }
        ]


class ExpectedAgentBehaviors:
    """Expected behaviors for different agent types"""

    @staticmethod
    def random_agent_expectations():
        """Expected behavior patterns for RandomAgent"""
        return {
            'deterministic_with_seed': True,
            'respects_legal_actions': True,
            'action_distribution': 'uniform',  # Should be roughly uniform over legal actions
            'no_strategic_preference': True
        }

    @staticmethod
    def baseline_agent_expectations():
        """Expected behavior patterns for BaselineAgent"""
        return [
            {
                'scenario': 'high_probability_bid',
                'game_state': {
                    'dice_counts': [3, 3, 3],
                    'my_hand': [2, 2, 2],  # Have 3 twos
                    'current_bid': (2, 2),  # Bid 2 twos
                    'ones_are_wild': True
                },
                'expected_action_type': 'bid',  # Should raise, not call
                'explanation': 'High probability bid should be raised'
            },
            {
                'scenario': 'low_probability_bid',
                'game_state': {
                    'dice_counts': [3, 3, 3],
                    'my_hand': [4, 5, 6],  # No twos or ones
                    'current_bid': (6, 2),  # Bid 6 twos (impossible)
                    'ones_are_wild': True
                },
                'expected_action_type': 'call',  # Should call impossible bid
                'explanation': 'Low probability bid should be called'
            }
        ]

    @staticmethod
    def mc_agent_expectations():
        """Expected behavior patterns for MonteCarloAgent"""
        return {
            'better_than_random': True,
            'simulation_based_decisions': True,
            'parameter_sensitivity': {
                'n_simulations': 'higher_n_better_decisions',
                'chunk_size': 'affects_performance_not_quality',
                'early_stop_margin': 'affects_computation_time'
            },
            'expected_performance_order': ['mc', 'baseline', 'random']  # Best to worst
        }


class ExpectedBidValidations:
    """Expected bid validation results"""

    @staticmethod
    def valid_bid_progressions():
        """Valid bid progression examples"""
        return [
            {
                'progression': [(1, 2), (1, 3), (2, 3), (2, 4)],
                'all_valid': True,
                'explanation': 'Normal progression: face increase, then quantity increase'
            },
            {
                'progression': [(2, 1), (5, 2)],  # Ones to regular
                'all_valid': True,
                'explanation': 'Valid transition from ones to regular face'
            },
            {
                'progression': [(4, 6), (2, 1)],  # Regular to ones
                'all_valid': True,
                'explanation': 'Valid transition from regular face to ones'
            }
        ]

    @staticmethod
    def invalid_bid_progressions():
        """Invalid bid progression examples"""
        return [
            {
                'progression': [(2, 3), (1, 3)],  # Quantity decrease
                'all_valid': False,
                'explanation': 'Cannot decrease quantity on same face'
            },
            {
                'progression': [(2, 4), (2, 3)],  # Face decrease
                'all_valid': False,
                'explanation': 'Cannot decrease face with same quantity'
            },
            {
                'progression': [(3, 1), (3, 2)],  # Invalid ones transition
                'all_valid': False,
                'explanation': 'Invalid transition from ones to regular face'
            }
        ]


class ExpectedPerformanceMetrics:
    """Expected performance characteristics"""

    @staticmethod
    def mc_agent_performance_expectations():
        """Expected performance metrics for MC agent with different parameters"""
        return {
            'mc_n_100_vs_1000': {
                'speed_improvement': 'significant',  # Should be much faster
                'decision_quality': 'acceptable',    # Should maintain reasonable quality
                'memory_usage': 'lower'              # Should use less memory
            },
            'tournament_completion_time': {
                'with_mc_n_100': 'under_30_seconds',  # For small tournaments
                'with_mc_n_1000': 'over_2_minutes',   # Significantly slower
                'improvement_factor': 'at_least_4x'    # At least 4x faster
            }
        }

    @staticmethod
    def scalability_expectations():
        """Expected scalability characteristics"""
        return {
            'players_2_to_6': 'linear_scaling',
            'games_10_to_100': 'linear_scaling',
            'memory_usage': 'stable_over_time',
            'max_reasonable_tournament_size': {
                'players': 6,
                'games_per_matchup': 100,
                'total_time_limit': '10_minutes'
            }
        }


class ExpectedRegressionResults:
    """Expected results for regression testing"""

    @staticmethod
    def parameter_change_validation():
        """Expected validation results for MC_N parameter change"""
        return {
            'mc_n_reduction_impact': {
                'decision_time': 'significantly_reduced',
                'decision_quality': 'maintained_within_acceptable_range'
            },
            'backward_compatibility': {
                'agent_interfaces': 'unchanged',
                'tournament_system': 'fully_compatible',
                'game_mechanics': 'identical_behavior'
            }
        }

    @staticmethod
    def quality_thresholds():
        """Quality thresholds that must be maintained"""
        return {
            'decision_time_per_action': 1.0,      # Should decide within 1 second per action
            'tournament_completion_rate': 1.0     # All tournaments should complete successfully
        }
