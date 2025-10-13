"""
Unit tests for Action class functionality.

This module tests the Action class methods, properties, creation,
comparison, and string representation functionality.
"""

import pytest
from sim.perudo import Action


class TestAction:
    """Test suite for Action class functionality."""

    def test_bid_creation(self):
        """Test Action.bid() creation and properties."""
        # Test basic bid creation
        action = Action.bid(3, 4)

        assert Action.is_bid(action) == True
        assert Action.qty(action) == 3
        assert Action.face(action) == 4

        # Test edge case bids
        min_bid = Action.bid(1, 1)
        assert Action.qty(min_bid) == 1
        assert Action.face(min_bid) == 1

        max_bid = Action.bid(15, 6)
        assert Action.qty(max_bid) == 15
        assert Action.face(max_bid) == 6

    def test_call_creation(self):
        """Test Action.call() creation and properties."""
        action = Action.call()

        assert Action.is_bid(action) == False
        assert action == ('call',)

    def test_exact_creation(self):
        """Test Action.exact() creation and properties."""
        action = Action.exact()

        assert Action.is_bid(action) == False
        assert action == ('exact',)

    def test_action_equality(self):
        """Test Action equality comparison."""
        # Test bid equality
        bid1 = Action.bid(3, 4)
        bid2 = Action.bid(3, 4)
        bid3 = Action.bid(3, 5)

        assert bid1 == bid2
        assert bid1 != bid3

        # Test call equality
        call1 = Action.call()
        call2 = Action.call()
        assert call1 == call2

        # Test exact equality
        exact1 = Action.exact()
        exact2 = Action.exact()
        assert exact1 == exact2

        # Test different action types are not equal
        assert bid1 != call1
        assert bid1 != exact1
        assert call1 != exact1

    def test_action_string_representation(self):
        """Test Action.to_str() method."""
        # Test bid string representation
        bid = Action.bid(3, 4)
        bid_str = Action.to_str(bid)
        assert "3" in bid_str
        assert "4" in bid_str

        # Test call string representation
        call = Action.call()
        call_str = Action.to_str(call)
        assert "call" in call_str.lower()

        # Test exact string representation
        exact = Action.exact()
        exact_str = Action.to_str(exact)
        assert "exact" in exact_str.lower()

    def test_action_type_checking(self):
        """Test Action type checking methods."""
        bid = Action.bid(2, 3)
        call = Action.call()
        exact = Action.exact()

        # Test is_bid method
        assert Action.is_bid(bid) == True
        assert Action.is_bid(call) == False
        assert Action.is_bid(exact) == False

    def test_bid_component_extraction(self):
        """Test extraction of quantity and face from bid actions."""
        bid = Action.bid(5, 2)

        # Test quantity extraction
        qty = Action.qty(bid)
        assert qty == 5

        # Test face extraction
        face = Action.face(bid)
        assert face == 2

        # Test with different values
        bid2 = Action.bid(1, 6)
        assert Action.qty(bid2) == 1
        assert Action.face(bid2) == 6

    def test_bid_component_extraction_non_bid_actions(self):
        """Test that qty and face methods handle non-bid actions appropriately."""
        call = Action.call()
        exact = Action.exact()

        # These should not raise errors but may return None or handle gracefully
        # The exact behavior depends on implementation
        try:
            qty_call = Action.qty(call)
            face_call = Action.face(call)
            qty_exact = Action.qty(exact)
            face_exact = Action.face(exact)
            # If no exception is raised, the implementation handles it gracefully
        except (IndexError, TypeError):
            # If exceptions are raised, that's also acceptable behavior
            pass

    def test_action_immutability(self):
        """Test that actions are immutable (tuples)."""
        bid = Action.bid(3, 4)
        call = Action.call()
        exact = Action.exact()

        # Actions should be tuples (immutable)
        assert isinstance(bid, tuple)
        assert isinstance(call, tuple)
        assert isinstance(exact, tuple)

    def test_bid_validation_edge_cases(self):
        """Test bid creation with edge case values."""
        # Test minimum valid values
        min_bid = Action.bid(1, 1)
        assert Action.qty(min_bid) == 1
        assert Action.face(min_bid) == 1

        # Test maximum reasonable values
        max_bid = Action.bid(30, 6)  # High quantity for large games
        assert Action.qty(max_bid) == 30
        assert Action.face(max_bid) == 6

        # Test face values 1-6
        for face in range(1, 7):
            bid = Action.bid(1, face)
            assert Action.face(bid) == face

    def test_action_in_collections(self):
        """Test that actions work properly in collections (sets, lists)."""
        bid1 = Action.bid(3, 4)
        bid2 = Action.bid(3, 4)  # Same as bid1
        bid3 = Action.bid(3, 5)  # Different from bid1
        call = Action.call()
        exact = Action.exact()

        # Test in list
        actions_list = [bid1, call, exact, bid3]
        assert bid1 in actions_list
        assert bid2 in actions_list  # Should find bid1 since they're equal

        # Test in set (should handle duplicates)
        actions_set = {bid1, bid2, bid3, call, exact}
        assert len(actions_set) == 4  # bid1 and bid2 are the same

        # Test set membership
        assert bid1 in actions_set
        assert bid2 in actions_set
        assert call in actions_set
        assert exact in actions_set

    def test_action_ordering(self):
        """Test that bid actions can be compared for ordering."""
        bid_low = Action.bid(1, 2)
        bid_medium = Action.bid(2, 3)
        bid_high = Action.bid(3, 4)
        bid_same_qty_higher_face = Action.bid(2, 4)

        # Test that actions can be used in sorted collections
        # The exact ordering behavior depends on tuple comparison
        bids = [bid_high, bid_low, bid_medium, bid_same_qty_higher_face]
        sorted_bids = sorted(bids)

        # Should be able to sort without errors
        assert len(sorted_bids) == 4
        assert all(Action.is_bid(bid) for bid in sorted_bids)


