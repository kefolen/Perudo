import pytest

from web.interactive_game import InteractivePerudoGame


def make_game(num_humans=3, num_ai=0):
    players = [f"P{i}" for i in range(num_humans)]
    ai_agents = []
    return InteractivePerudoGame(players, ai_agents, auto_continue_delay=0)


def test_maputa_triggers_only_for_previous_round_loser_with_one_die():
    game = make_game(3)

    # Set up dice counts so that player 1 will lose from 2 -> 1
    game.state['dice_counts'] = [3, 2, 2]
    game.current_player = 1

    # End the round with player 1 as the loser
    game._handle_round_end(loser=1, actual_count=0)
    # Directly continue to start a new round
    game._continue_after_round_end()

    # Player 1 should start and maputa should be active because they just dropped to 1
    assert game.current_player == 1
    assert game.state['dice_counts'][1] == 1
    assert game.maputa_active is True


def test_maputa_does_not_trigger_when_loser_eliminated_and_next_starter_has_one_die():
    game = make_game(3)

    # Player order: 0,1,2. Player 1 will be eliminated (1->0). Player 2 has one die already.
    game.state['dice_counts'] = [3, 1, 1]
    game.current_player = 1

    # End the round with player 1 as the loser (eliminated)
    game._handle_round_end(loser=1, actual_count=0)
    game._continue_after_round_end()

    # Next alive player (2) should start, but maputa should NOT trigger for them
    assert game.state['dice_counts'][1] == 0
    assert game.current_player == 2
    assert game.state['dice_counts'][2] == 1
    assert game.maputa_active is False


def test_maputa_applies_only_to_immediate_next_round():
    game = make_game(3)

    # Player 0 loses from 2 -> 1 and should trigger maputa for the next round
    game.state['dice_counts'] = [2, 3, 3]
    game.current_player = 0

    game._handle_round_end(loser=0, actual_count=0)
    game._continue_after_round_end()

    # First next round: active
    assert game.current_player == 0
    assert game.state['dice_counts'][0] == 1
    assert game.maputa_active is True

    # Start another new round without a prior loss marking player 0
    game._start_new_round()

    # Even if player 0 still has one die, maputa should now be False because
    # it should only apply immediately after the loss that reduced to one die
    assert game.state['dice_counts'][0] == 1
    assert game.maputa_active is False
