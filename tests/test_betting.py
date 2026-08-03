import pytest

from poker.betting import Action, BettingEngine, BettingPlayer


def test_player_commit_reduces_stack_and_adds_contribution():
    player = BettingPlayer("Alice", 100)

    committed = player.commit(25)

    assert committed == 25
    assert player.stack == 75
    assert player.contribution == 25


def test_fold_marks_player_as_folded():
    player = BettingPlayer("Alice", 100)
    engine = BettingEngine([player], minimum_bet=10)

    engine.fold(player)

    assert player.folded is True


def test_check_is_allowed_when_nothing_is_to_call():
    player = BettingPlayer("Alice", 100)
    engine = BettingEngine([player], minimum_bet=10)

    engine.check(player)

    assert engine.state.pot == 0


def test_check_is_rejected_when_facing_a_bet():
    alice = BettingPlayer("Alice", 100)
    bob = BettingPlayer("Bob", 100)
    engine = BettingEngine([alice, bob], minimum_bet=10)

    engine.bet(alice, 20)

    with pytest.raises(ValueError, match="cannot check"):
        engine.check(bob)


def test_call_matches_current_bet():
    alice = BettingPlayer("Alice", 100)
    bob = BettingPlayer("Bob", 100)
    engine = BettingEngine([alice, bob], minimum_bet=10)

    engine.bet(alice, 20)
    committed = engine.call(bob)

    assert committed == 20
    assert bob.stack == 80
    assert bob.contribution == 20
    assert engine.state.pot == 40


def test_bet_sets_current_bet():
    player = BettingPlayer("Alice", 100)
    engine = BettingEngine([player], minimum_bet=10)

    committed = engine.bet(player, 30)

    assert committed == 30
    assert player.stack == 70
    assert engine.state.pot == 30
    assert engine.state.current_bet == 30


def test_bet_below_minimum_is_rejected():
    player = BettingPlayer("Alice", 100)
    engine = BettingEngine([player], minimum_bet=10)

    with pytest.raises(ValueError, match="bet must be at least"):
        engine.bet(player, 5)


def test_raise_updates_current_bet():
    alice = BettingPlayer("Alice", 100)
    bob = BettingPlayer("Bob", 100)
    engine = BettingEngine([alice, bob], minimum_bet=10)

    engine.bet(alice, 20)
    engine.raise_bet(bob, 40)

    assert bob.contribution == 40
    assert bob.stack == 60
    assert engine.state.pot == 60
    assert engine.state.current_bet == 40


def test_all_in_commits_entire_stack():
    player = BettingPlayer("Alice", 75)
    engine = BettingEngine([player], minimum_bet=10)

    committed = engine.all_in(player)

    assert committed == 75
    assert player.stack == 0
    assert player.contribution == 75
    assert player.all_in is True
    assert engine.state.pot == 75


def test_folded_player_cannot_act():
    player = BettingPlayer("Alice", 100)
    engine = BettingEngine([player], minimum_bet=10)

    engine.fold(player)

    with pytest.raises(ValueError, match="folded player"):
        engine.check(player)


def test_action_enum_contains_supported_actions():
    assert Action.FOLD.value == "fold"
    assert Action.CHECK.value == "check"
    assert Action.CALL.value == "call"
    assert Action.BET.value == "bet"
    assert Action.RAISE.value == "raise"
    assert Action.ALL_IN.value == "all_in"
