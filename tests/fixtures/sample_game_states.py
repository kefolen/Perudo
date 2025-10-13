"""
Sample game states for testing various scenarios in Perudo.

This module provides predefined game states covering different phases of the game,
special rule scenarios, and edge cases for comprehensive testing.
"""

from sim.perudo import Action


class GameStates:
    """Collection of sample game states for testing"""
    
    @staticmethod
    def early_game_no_bid():
        """Early game state with no current bid - 5 dice per player"""
        return {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [1, 2, 3, 4, 5],
            'current_bid': None,
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def early_game_with_bid():
        """Early game state with an existing bid"""
        return {
            'dice_counts': [5, 5, 5],
            'player_idx': 1,
            'my_hand': [2, 2, 3, 4, 6],
            'current_bid': (2, 3),
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def mid_game_balanced():
        """Mid game state with balanced dice counts"""
        return {
            'dice_counts': [3, 2, 3],
            'player_idx': 0,
            'my_hand': [1, 4, 6],
            'current_bid': (3, 2),
            'maputa_restrict_face': None,
            'round_num': 3
        }
    
    @staticmethod
    def mid_game_unbalanced():
        """Mid game state with unbalanced dice distribution"""
        return {
            'dice_counts': [4, 1, 2],
            'player_idx': 2,
            'my_hand': [5, 5],
            'current_bid': (2, 1),
            'maputa_restrict_face': None,
            'round_num': 4
        }
    
    @staticmethod
    def end_game_single_dice():
        """End game state where all players have single dice"""
        return {
            'dice_counts': [1, 1, 1],
            'player_idx': 0,
            'my_hand': [3],
            'current_bid': (1, 2),
            'maputa_restrict_face': None,
            'round_num': 8
        }
    
    @staticmethod
    def end_game_mixed_dice():
        """End game state with mixed dice counts"""
        return {
            'dice_counts': [2, 1, 1],
            'player_idx': 1,
            'my_hand': [6],
            'current_bid': (1, 4),
            'maputa_restrict_face': None,
            'round_num': 7
        }
    
    @staticmethod
    def maputa_restriction_scenario():
        """Game state with maputa restriction active"""
        return {
            'dice_counts': [3, 1, 2],
            'player_idx': 1,
            'my_hand': [4],
            'current_bid': (1, 3),
            'maputa_restrict_face': 3,  # Player with 1 die bid on 3s
            'round_num': 6
        }
    
    @staticmethod
    def high_bid_scenario():
        """Game state with a high bid requiring careful consideration"""
        return {
            'dice_counts': [4, 3, 4],
            'player_idx': 2,
            'my_hand': [2, 2, 6, 6],
            'current_bid': (7, 2),
            'maputa_restrict_face': None,
            'round_num': 2
        }
    
    @staticmethod
    def ones_bid_scenario():
        """Game state with a bid on ones (special wild card rules)"""
        return {
            'dice_counts': [3, 3, 2],
            'player_idx': 0,
            'my_hand': [1, 1, 4],
            'current_bid': (2, 1),
            'maputa_restrict_face': None,
            'round_num': 5
        }
    
    @staticmethod
    def all_ones_hand():
        """Game state where player has all ones in hand"""
        return {
            'dice_counts': [4, 4, 3],
            'player_idx': 1,
            'my_hand': [1, 1, 1, 1],
            'current_bid': (3, 5),
            'maputa_restrict_face': None,
            'round_num': 2
        }
    
    @staticmethod
    def no_ones_hand():
        """Game state where player has no ones in hand"""
        return {
            'dice_counts': [5, 4, 4],
            'player_idx': 2,
            'my_hand': [2, 3, 4, 5],
            'current_bid': (4, 2),
            'maputa_restrict_face': None,
            'round_num': 1
        }
    
    @staticmethod
    def maximum_bid_scenario():
        """Game state approaching maximum possible bid"""
        return {
            'dice_counts': [5, 5, 5],
            'player_idx': 0,
            'my_hand': [6, 6, 6, 6, 6],
            'current_bid': (12, 6),
            'maputa_restrict_face': None,
            'round_num': 1
        }


class BidProgressions:
    """Sample bid progression scenarios for testing"""
    
    @staticmethod
    def normal_progression():
        """Normal bid progression from low to medium"""
        return [
            (1, 2),  # Starting bid
            (1, 3),  # Face increase
            (2, 3),  # Quantity increase
            (2, 4),  # Face increase
            (3, 4),  # Quantity increase
        ]
    
    @staticmethod
    def ones_to_regular():
        """Bid progression from ones to regular faces"""
        return [
            (2, 1),  # Bid on ones
            (5, 2),  # Switch to regular face (2*2+1 = 5)
            (5, 3),  # Face increase
            (6, 3),  # Quantity increase
        ]
    
    @staticmethod
    def regular_to_ones():
        """Bid progression from regular faces to ones"""
        return [
            (4, 3),  # Regular bid
            (4, 4),  # Face increase
            (4, 5),  # Face increase
            (4, 6),  # Face increase
            (2, 1),  # Switch to ones (4//2 = 2)
        ]
    
    @staticmethod
    def escalating_quantities():
        """Rapidly escalating quantity bids"""
        return [
            (1, 2),
            (3, 2),
            (5, 2),
            (7, 2),
            (9, 2),
        ]


class SpecialScenarios:
    """Special game scenarios for edge case testing"""
    
    @staticmethod
    def two_player_endgame():
        """Two players remaining, both with single dice"""
        return {
            'dice_counts': [1, 0, 1],  # Middle player eliminated
            'player_idx': 0,
            'my_hand': [2],
            'current_bid': None,
            'maputa_restrict_face': None,
            'round_num': 10
        }
    
    @staticmethod
    def player_about_to_be_eliminated():
        """Player with 1 die facing a challenging bid"""
        return {
            'dice_counts': [1, 4, 3],
            'player_idx': 0,
            'my_hand': [5],
            'current_bid': (4, 3),
            'maputa_restrict_face': None,
            'round_num': 8
        }
    
    @staticmethod
    def exact_call_opportunity():
        """Scenario where exact call might be beneficial"""
        return {
            'dice_counts': [2, 2, 2],
            'player_idx': 1,
            'my_hand': [3, 3],
            'current_bid': (3, 3),  # Exactly 3 threes might be on table
            'maputa_restrict_face': None,
            'round_num': 6
        }
    
    @staticmethod
    def impossible_bid():
        """Scenario with clearly impossible bid"""
        return {
            'dice_counts': [2, 1, 1],
            'player_idx': 2,
            'my_hand': [4],
            'current_bid': (10, 6),  # Impossible with only 4 dice total
            'maputa_restrict_face': None,
            'round_num': 9
        }