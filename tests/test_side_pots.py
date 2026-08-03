import pytest

from agents.base_agent import PokerAgent
from poker.betting import BettingEngine, BettingPlayer
from poker.card import Card
from poker.game import HoldemGame
from poker.player import Player


def card(rank, suit):
    suits = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}
    return Card(rank, suits[suit])


class FixedDeck:
    def __init__(self, cards):
        self._cards = list(cards)

    def draw_many(self, count):
        if count > len(self._cards):
            raise RuntimeError("fixed deck exhausted")
        dealt, self._cards = self._cards[:count], self._cards[count:]
        return dealt


class CheckCallAgent(PokerAgent):
    """Check whenever possible, otherwise call."""

    def decide(self, context):
        if "check" in context.allowed_actions:
            return "check"
        return "call"


class AllInAgent(PokerAgent):
    def decide(self, context):
        return "all_in"


class ScriptedAgent(PokerAgent):
    """Return actions from a queue in order."""

    def __init__(self, actions):
        self.actions = list(actions)

    def decide(self, context):
        if not self.actions:
            raise RuntimeError("script exhausted")
        return self.actions.pop(0)


# ----------------------------------------------------------------------
# BettingState.compute_pots
# ----------------------------------------------------------------------


def test_compute_pots_equal_contributions():
    engine = BettingEngine(
        [
            BettingPlayer("A", 100),
            BettingPlayer("B", 100),
            BettingPlayer("C", 100),
        ],
        minimum_bet=10,
    )

    for player in engine.state.players:
        engine.state.add_to_pot(player, 10)

    pots = engine.state.compute_pots()

    assert pots == [(30, engine.state.players)]


def test_compute_pots_three_levels():
    engine = BettingEngine(
        [
            BettingPlayer("A", 200),
            BettingPlayer("B", 200),
            BettingPlayer("C", 200),
        ],
        minimum_bet=10,
    )
    a, b, c = engine.state.players

    engine.state.add_to_pot(a, 50)
    engine.state.add_to_pot(b, 100)
    engine.state.add_to_pot(c, 200)

    pots = engine.state.compute_pots()

    assert len(pots) == 3

    amount, eligible = pots[0]
    assert amount == 150
    assert eligible == [a, b, c]

    amount, eligible = pots[1]
    assert amount == 100
    assert eligible == [b, c]

    amount, eligible = pots[2]
    assert amount == 100
    assert eligible == [c]


def test_compute_pots_folded_player_is_not_eligible():
    engine = BettingEngine(
        [
            BettingPlayer("A", 100),
            BettingPlayer("B", 100),
            BettingPlayer("C", 100),
        ],
        minimum_bet=10,
    )
    a, b, c = engine.state.players

    for player in (a, b, c):
        engine.state.add_to_pot(player, 10)

    engine.fold(a)

    assert engine.state.compute_pots() == [(30, [b, c])]


def test_compute_pots_short_all_in():
    engine = BettingEngine(
        [BettingPlayer("A", 5), BettingPlayer("B", 10)],
        minimum_bet=10,
    )
    a, b = engine.state.players

    engine.state.add_to_pot(a, 5)
    engine.state.add_to_pot(b, 10)

    assert engine.state.compute_pots() == [(10, [a, b]), (5, [b])]


def test_total_contribution_accumulates_across_streets():
    engine = BettingEngine(
        [BettingPlayer("A", 100), BettingPlayer("B", 100)],
        minimum_bet=10,
    )
    a, b = engine.state.players

    engine.state.add_to_pot(a, 20)
    engine.state.add_to_pot(b, 20)
    engine.reset_street()
    engine.state.add_to_pot(a, 10)
    engine.state.add_to_pot(b, 10)

    assert a.contribution == 10
    assert a.total_contribution == 30
    assert engine.state.compute_pots() == [(60, [a, b])]


# ----------------------------------------------------------------------
# game-level side pot integration
# ----------------------------------------------------------------------


def test_distinct_main_and_side_pot_winners(monkeypatch):
    deck = [
        card(14, "c"), card(14, "d"),  # A: pair of aces
        card(13, "c"), card(13, "d"),  # B: pair of kings
        card(12, "c"), card(12, "d"),  # C: pair of queens
        card(7, "d"), card(8, "d"), card(9, "d"),
        card(2, "s"), card(3, "s"),
    ]
    monkeypatch.setattr("poker.game.Deck", lambda seed: FixedDeck(deck))

    players = [
        Player("A", AllInAgent(), chips=10),  # button, short all-in
        Player(
            "B",
            ScriptedAgent(["raise", "check", "check", "check"]),
            chips=100,
        ),  # small blind
        Player("C", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    # A is all-in for 10; B raises to 20 and C calls.
    # Main pot 10 * 3 = 30 (A wins), side pot 10 * 2 = 20 (B wins).
    assert [winner.name for winner in winners] == ["A"]
    assert players[0].chips == 30  # 0 + main pot
    assert players[1].chips == 100  # 80 + side pot
    assert players[2].chips == 80  # 80 + nothing
    assert game.pot == 50
    assert sum(player.chips for player in players) == 210


def test_all_in_street_skips_all_in_player():
    players = [
        Player("A", AllInAgent(), chips=10),  # button, short all-in
        Player("B", CheckCallAgent(), chips=100),  # small blind
        Player("C", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert players[0].all_in
    assert len(game.community) == 5
    assert game.pot == 30  # 10 from each player
    assert sum(player.chips for player in players) == 210


def test_showdown_with_zero_contribution_returns_winner():
    players = [
        Player("A", CheckCallAgent(), chips=0),
        Player("B", CheckCallAgent(), chips=0),
        Player("C", CheckCallAgent(), chips=0),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert game.pot == 0
    assert sum(player.chips for player in players) == 0


def test_side_pot_chips_are_conserved():
    players = [
        Player("A", AllInAgent(), chips=7),
        Player("B", CheckCallAgent(), chips=100),
        Player("C", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    # A all-in 7, B and C put in 10 each: 21 main pot, 6 side pot.
    assert game.pot == 27
    assert sum(player.chips for player in players) == 207
