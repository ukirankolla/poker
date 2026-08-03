import json

import pytest

from agents.base_agent import PokerAgent
from agents.ollama_agent import OllamaAgent
from poker.betting import BettingEngine, BettingPlayer
from poker.card import Card
from poker.game import HoldemGame
from poker.player import Player


class CheckCallAgent(PokerAgent):
    """Check whenever possible, otherwise call."""

    def decide(self, context):
        if "check" in context.allowed_actions:
            return "check"
        return "call"


class FoldAgent(PokerAgent):
    def decide(self, context):
        return "fold"


class AllInAgent(PokerAgent):
    def decide(self, context):
        return "all_in"


class IllegalAgent(PokerAgent):
    def decide(self, context):
        return "raise"


class ScriptedAgent(PokerAgent):
    """Return actions from a queue in order."""

    def __init__(self, actions):
        self.actions = list(actions)

    def decide(self, context):
        if not self.actions:
            raise RuntimeError("script exhausted")
        return self.actions.pop(0)


class RecordingAgent(PokerAgent):
    """Record the community size at every decision point."""

    def __init__(self):
        self.snapshots = []

    def decide(self, context):
        self.snapshots.append(len(context.community_cards))
        return "check" if "check" in context.allowed_actions else "call"


def card(rank, suit):
    suits = {"c": "clubs", "d": "diamonds", "h": "hearts", "s": "spades"}
    return Card(rank, suits[suit])


def kkk_qq_deck(player_count):
    """Low, unrelated hole cards for each player plus a KKKQQ board.

    The board plays as a full house for everyone, so all players tie
    at showdown regardless of their hole cards.
    """
    holes = [
        (2, "c"), (3, "c"),
        (4, "d"), (5, "d"),
        (6, "h"), (7, "h"),
        (8, "s"), (9, "s"),
        (10, "s"), (11, "s"),
    ]
    board = [
        (13, "c"), (13, "d"), (13, "h"),
        (12, "c"), (12, "d"),
    ]
    cards = []
    for index in range(player_count):
        cards.extend(
            card(rank, suit)
            for rank, suit in holes[index * 2: index * 2 + 2]
        )
    cards.extend(card(rank, suit) for rank, suit in board)
    return cards


class FixedDeck:
    def __init__(self, cards):
        self._cards = list(cards)

    def draw_many(self, count):
        if count > len(self._cards):
            raise RuntimeError("fixed deck exhausted")
        dealt, self._cards = self._cards[:count], self._cards[count:]
        return dealt


# ----------------------------------------------------------------------
# blinds and button rotation
# ----------------------------------------------------------------------


def test_blinds_posted_before_preflop():
    players = [
        Player("A", ScriptedAgent(["fold"]), chips=100),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1, small_blind=5, big_blind=10)

    winners, score = game.play_hand()

    assert winners[0].name == "B"
    assert score == (9,)  # everyone else folded
    assert game.pot == 15  # small blind 5 + big blind 10
    assert players[0].chips == 95  # only lost the small blind
    assert players[1].chips == 105  # 100 - big blind + pot


def test_button_rotates_between_hands():
    players = [
        Player("A", CheckCallAgent()),
        Player("B", CheckCallAgent()),
        Player("C", CheckCallAgent()),
    ]

    game = HoldemGame(players, seed=1)

    assert game.button_index == -1

    for expected in (0, 1, 2, 0, 1):
        game.play_hand()
        assert game.button_index == expected


def test_blind_positions_rotate():
    players = [
        Player("A", CheckCallAgent()),
        Player("B", CheckCallAgent()),
        Player("C", CheckCallAgent()),
    ]

    game = HoldemGame(players)

    game.button_index = 0
    assert game._blind_positions() == (1, 2)
    game.button_index = 1
    assert game._blind_positions() == (2, 0)
    game.button_index = 2
    assert game._blind_positions() == (0, 1)


def test_heads_up_button_posts_small_blind():
    players = [
        Player("A", CheckCallAgent()),
        Player("B", CheckCallAgent()),
    ]

    game = HoldemGame(players)

    game.button_index = 0
    assert game._blind_positions() == (0, 1)
    game.button_index = 1
    assert game._blind_positions() == (1, 0)


# ----------------------------------------------------------------------
# street progression
# ----------------------------------------------------------------------


def test_street_progression_and_deal_counts():
    players = [
        Player("A", RecordingAgent(), chips=1000),
        Player("B", RecordingAgent(), chips=1000),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    assert len(game.community) == 5

    streets = {entry["street"] for entry in game.action_history}
    assert streets == {"preflop", "flop", "turn", "river"}

    # preflop: 0 community cards, then 3, 4, 5 as streets are dealt.
    for player in players:
        assert sorted(set(player.agent.snapshots)) == [0, 3, 4, 5]


def test_fold_on_flop_wins_by_fold():
    players = [
        Player("A", ScriptedAgent(["call", "fold"]), chips=100),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners[0].name == "B"
    assert len(game.community) == 3  # flop dealt, no turn or river
    assert game.pot == 20  # SB 5 + call 5 + BB 10
    assert players[0].chips == 90
    assert players[1].chips == 110


# ----------------------------------------------------------------------
# showdown, pot award, splits, odd chip
# ----------------------------------------------------------------------


def test_showdown_awards_pot_and_conserves_chips():
    players = [
        Player("A", CheckCallAgent(), chips=100),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, score = game.play_hand()

    assert len(winners) == 1
    assert score[0] >= 0
    assert game.pot == 20
    assert sorted(player.chips for player in players) == [90, 110]
    assert sum(player.chips for player in players) == 200


def test_split_pot_when_players_tie(monkeypatch):
    monkeypatch.setattr(
        "poker.game.Deck",
        lambda seed: FixedDeck(kkk_qq_deck(2)),
    )

    players = [
        Player("A", CheckCallAgent(), chips=100),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert len(winners) == 2
    assert game.pot == 20
    assert sum(player.chips for player in players) == 200
    # Each player contributed 10 and gets exactly 10 back.
    assert players[0].chips == 100
    assert players[1].chips == 100


def test_short_all_in_creates_side_pot(monkeypatch):
    monkeypatch.setattr(
        "poker.game.Deck",
        lambda seed: FixedDeck(kkk_qq_deck(3)),
    )

    players = [
        Player("A", AllInAgent(), chips=8),  # button, all-in for 8
        Player("B", ScriptedAgent(["call"]), chips=10),  # small blind
        Player("C", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert len(winners) == 3
    # Contributions are 8 + 10 + 10 = 28.
    # Main pot is 8 * 3 = 24 split three ways; side pot is 2 * 2 = 4
    # split between B and C only.
    assert game.pot == 28
    assert players[0].chips == 8  # 0 + 8 (main pot share only)
    assert players[1].chips == 10  # 0 + 8 (main) + 2 (side)
    assert players[2].chips == 100  # 90 + 8 (main) + 2 (side)
    assert sum(player.chips for player in players) == 118


def test_odd_chip_goes_to_earliest_winner(monkeypatch):
    monkeypatch.setattr(
        "poker.game.Deck",
        lambda seed: FixedDeck(kkk_qq_deck(3)),
    )

    players = [
        Player("A", ScriptedAgent(["call", "check", "check", "check"]), chips=100),  # button
        Player("B", ScriptedAgent(["fold"]), chips=100),  # small blind
        Player("C", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert len(winners) == 2
    # A and C tie on KKKQQ; B folds after posting the small blind 5.
    # Pot is 10 + 5 + 10 = 25; divmod(25, 2) -> 12 remainder 1.
    assert game.pot == 25
    assert players[0].chips == 103  # 90 + 12 + odd chip
    assert players[1].chips == 95  # folded away the small blind
    assert players[2].chips == 102  # 90 + 12
    assert sum(player.chips for player in players) == 300


# ----------------------------------------------------------------------
# agent actions
# ----------------------------------------------------------------------


def test_agent_actions_recorded_with_streets():
    players = [
        Player("A", CheckCallAgent(), chips=1000),
        Player("B", CheckCallAgent(), chips=1000),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    actions = [entry["action"] for entry in game.action_history]
    streets = [entry["street"] for entry in game.action_history]

    assert "call" in actions  # preflop small blind completion
    assert actions.count("check") >= 3
    assert streets[0] == "preflop"
    assert {"flop", "turn", "river"} <= set(streets)


def test_illegal_agent_action_raises():
    players = [
        Player("A", IllegalAgent(), chips=12),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    with pytest.raises(ValueError, match="illegal action"):
        game.play_hand()


def test_insufficient_stack_becomes_all_in():
    players = [
        Player("A", ScriptedAgent(["call"]), chips=8),
        Player("B", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    assert players[0].all_in is True
    assert game.pot == 18  # A all-in for 8 + B's big blind 10
    assert sum(player.chips for player in players) == 108


def test_everyone_all_in_runs_to_showdown():
    players = [
        Player("A", AllInAgent(), chips=50),
        Player("B", AllInAgent(), chips=50),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert len(game.community) == 5
    assert game.pot == 100


def test_raise_war_preflop():
    players = [
        Player(
            "A",
            ScriptedAgent(["call", "raise", "check", "check", "check"]),
            chips=100,
        ),
        Player(
            "B",
            ScriptedAgent(["raise", "call", "check", "check", "check"]),
            chips=100,
        ),
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    # Both end up contributing 30; the pot is 60.
    assert game.pot == 60
    assert sorted(player.chips for player in players) == [70, 130]
    assert sum(player.chips for player in players) == 200

    raises = [
        entry for entry in game.action_history if entry["action"] == "raise"
    ]
    assert len(raises) == 2
    assert all(entry["street"] == "preflop" for entry in raises)


def test_ollama_agent_plays_in_game(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"response": json.dumps({"action": "call"})}

    monkeypatch.setattr(
        "agents.ollama_agent.requests.post",
        lambda *args, **kwargs: FakeResponse(),
    )

    players = [
        Player("A", OllamaAgent(), chips=100),
        Player("B", OllamaAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert len(game.community) == 5


# ----------------------------------------------------------------------
# complete lifecycle
# ----------------------------------------------------------------------


def test_busted_big_blind_round_completes():
    """A busted big blind leaves current_bet below the small blind's
    contribution; the betting round must still terminate."""
    players = [
        Player("A", CheckCallAgent(), chips=100),
        Player("B", CheckCallAgent(), chips=100),
        Player("C", CheckCallAgent(), chips=0),  # busted big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert game.pot == 5  # only the small blind was posted
    assert sum(player.chips for player in players) == 200


def test_complete_hand_lifecycle():
    players = [
        Player("A", CheckCallAgent(), chips=100),
        Player("B", CheckCallAgent(), chips=100),
        Player("C", CheckCallAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    total = sum(player.chips for player in players)

    for hand in range(3):
        winners, _ = game.play_hand()

        assert winners
        assert game.button_index == hand % 3
        assert game.pot == 30  # every player calls the big blind
        assert len(game.community) == 5
        assert all(len(player.hole_cards) == 2 for player in players)
        assert game.action_history
        assert sum(player.chips for player in players) == total

        # State is reset for the next hand.
        assert all(not player.folded for player in players)
        assert all(player.current_bet == 0 for player in players)


def test_four_player_hand_lifecycle():
    players = [
        Player(name, CheckCallAgent(), chips=100)
        for name in ("A", "B", "C", "D")
    ]

    game = HoldemGame(players, seed=1)

    total = sum(player.chips for player in players)

    for hand in range(3):
        winners, _ = game.play_hand()

        assert winners
        assert game.button_index == hand % 4
        assert game.pot == 40  # every player calls the big blind
        assert len(game.community) == 5
        assert sum(player.chips for player in players) == total
        assert all(not player.folded for player in players)


def test_fold_after_raise():
    players = [
        Player(
            "A",
            ScriptedAgent(["raise", "fold"]),
            chips=100,
        ),  # button
        Player("B", CheckCallAgent(), chips=100),  # small blind
        Player("C", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    # A raises preflop to 20, then folds on the flop.
    assert players[0].folded
    assert game.pot == 60  # 20 from each player preflop
    assert players[0].chips == 80
    assert sum(player.chips for player in players) == 300
    assert {winner.name for winner in winners} <= {"B", "C"}
    assert len(game.community) == 5


def test_bet_then_call():
    players = [
        Player(
            "A",
            ScriptedAgent(["raise", "check", "check", "check"]),
            chips=100,
        ),  # button, posts the small blind
        Player("B", CheckCallAgent(), chips=100),  # big blind
    ]

    game = HoldemGame(players, seed=1)

    game.play_hand()

    # A raises preflop to 20, B calls to match; the pot is 40.
    assert game.pot == 40
    assert sorted(player.chips for player in players) == [80, 120]
    assert sum(player.chips for player in players) == 200

    actions = [entry["action"] for entry in game.action_history]
    assert "raise" in actions
    assert "call" in actions


# ----------------------------------------------------------------------
# engine-level helpers used by the game
# ----------------------------------------------------------------------


def test_reset_street_preserves_pot():
    engine = BettingEngine(
        [BettingPlayer("A", 100), BettingPlayer("B", 100)],
        minimum_bet=10,
    )

    engine.state.add_to_pot(engine.state.players[0], 20)
    engine.state.current_bet = 20

    pot_before = engine.state.pot

    engine.reset_street()

    assert engine.state.pot == pot_before
    assert engine.state.current_bet == 0
    assert engine.state.minimum_raise == 10
    assert all(player.contribution == 0 for player in engine.state.players)


def test_legal_actions_sets():
    game = HoldemGame(
        [Player("A", CheckCallAgent()), Player("B", CheckCallAgent())]
    )
    game._create_betting_engine()

    a, b = game._betting_players

    assert game._legal_actions(a) == ("fold", "check", "bet", "all_in")

    game._engine.state.add_to_pot(a, 10)
    game._engine.state.current_bet = 10

    b_actions = game._legal_actions(b)
    assert "check" not in b_actions
    assert "call" in b_actions
    assert "raise" in b_actions
    assert "all_in" in b_actions


def test_random_and_rule_based_agents_play_in_game():
    from agents.random_agent import RandomAgent
    from agents.rule_based_agent import RuleBasedAgent

    players = [
        Player("A", RandomAgent(seed=3), chips=100),
        Player("B", RuleBasedAgent(), chips=100),
        Player("C", FoldAgent(), chips=100),
    ]

    game = HoldemGame(players, seed=1)

    winners, _ = game.play_hand()

    assert winners
    assert sum(player.chips for player in players) == 300
    assert game.pot > 0
