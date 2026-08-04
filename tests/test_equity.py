from poker.card import Card
from poker.equity import estimate_equity, pot_odds


def test_aces_are_strong_preflop():
    equity = estimate_equity(
        (Card(14, "spades"), Card(14, "hearts")),
        num_opponents=1,
        trials=4000,
        seed=1,
    )

    assert equity > 0.8


def test_aces_stay_strong_against_many_opponents():
    equity = estimate_equity(
        (Card(14, "spades"), Card(14, "hearts")),
        num_opponents=3,
        trials=4000,
        seed=2,
    )

    assert 0.4 < equity < 0.9


def test_made_hand_on_river_is_near_certain():
    equity = estimate_equity(
        (Card(14, "spades"), Card(14, "hearts")),
        community_cards=(
            Card(14, "diamonds"),
            Card(14, "clubs"),
            Card(13, "spades"),
            Card(13, "hearts"),
            Card(12, "clubs"),
        ),
        num_opponents=1,
        trials=1000,
        seed=3,
    )

    assert equity == 1.0


def test_equity_is_deterministic_for_same_seed():
    hole = (Card(10, "spades"), Card(9, "hearts"))

    first = estimate_equity(hole, num_opponents=2, trials=2000, seed=7)
    second = estimate_equity(hole, num_opponents=2, trials=2000, seed=7)

    assert first == second


def test_no_opponents_is_always_won():
    equity = estimate_equity(
        (Card(2, "clubs"), Card(3, "diamonds")),
        num_opponents=0,
        trials=100,
        seed=5,
    )

    assert equity == 1.0


def test_invalid_trials_raise():
    try:
        estimate_equity((), num_opponents=1, trials=0)
    except ValueError:
        return

    raise AssertionError("trials=0 should raise ValueError")


def test_pot_odds():
    assert pot_odds(20, 80) == 0.2
    assert pot_odds(0, 100) == 0.0
