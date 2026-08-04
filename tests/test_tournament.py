import pytest

pytestmark = pytest.mark.integration

from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from poker.player import Player
from simulation.tournament import (
    TournamentConfig,
    TournamentResult,
    print_result,
    run_tournament,
)


def make_players(count):
    players = [Player(f"P{i}", RandomAgent(seed=i)) for i in range(count)]
    players += [
        Player(f"R{i}", RuleBasedAgent(seed=50 + i)) for i in range(count // 2)
    ]
    return players


@pytest.mark.regression
def test_single_table_tournament_conserves_chips():
    players = make_players(4)
    total = len(players) * 1000

    result = run_tournament(players, TournamentConfig(seed=7))

    assert result.winner is not None
    assert result.winner.name == result.standings[-1].name
    assert sum(player.chips for player in players) == total
    assert sum(player.chips for player in result.standings) == total


def test_multi_table_tournament_crowns_champion():
    players = make_players(20)
    total = len(players) * 1000

    result = run_tournament(
        players,
        TournamentConfig(
            starting_chips=1000,
            players_per_table=9,
            hands_per_level=10,
            seed=42,
        ),
    )

    assert result.winner is not None
    assert result.winner.chips > 0
    assert len(result.standings) == len(players)
    assert len({player.name for player in result.standings}) == len(players)
    assert sum(player.chips for player in players) == total


def test_tournament_is_deterministic_per_seed():
    first = run_tournament(make_players(8), TournamentConfig(seed=3))
    second = run_tournament(make_players(8), TournamentConfig(seed=3))

    assert [p.name for p in first.standings] == [p.name for p in second.standings]
    assert first.hands_played == second.hands_played
    assert first.winner.name == second.winner.name


def test_different_seed_gives_different_tournament():
    first = run_tournament(make_players(8), TournamentConfig(seed=1))
    second = run_tournament(make_players(8), TournamentConfig(seed=2))

    assert first.winner.name != second.winner.name


def test_blinds_escalate_as_levels_progress():
    players = make_players(8)
    result = run_tournament(
        players,
        TournamentConfig(
            starting_chips=1000,
            hands_per_level=10,
            seed=5,
        ),
    )

    assert result.hands_played >= 20
    assert result.level >= 2
    assert result.big_blind > 10


def test_prize_pool_payouts():
    players = make_players(6)

    result = run_tournament(
        players,
        TournamentConfig(prize_pool=10000, seed=9),
    )

    assert result.payouts
    assert sum(result.payouts.values()) == 10000
    assert result.payouts[result.standings[-1].name] == 5000
    assert result.payouts[result.standings[-2].name] == 3000
    assert result.payouts[result.standings[-3].name] == 2000


def test_prize_pool_none_gives_no_payouts():
    result = run_tournament(make_players(4), TournamentConfig(seed=1))

    assert result.payouts == {}


def test_requires_at_least_two_players():
    with pytest.raises(ValueError, match="at least two"):
        run_tournament([make_players(1)[0]])


def test_requires_positive_hands_per_level():
    with pytest.raises(ValueError, match="hands_per_level"):
        run_tournament(
            make_players(4), TournamentConfig(hands_per_level=0)
        )


def test_requires_two_players_per_table():
    with pytest.raises(ValueError, match="players_per_table"):
        run_tournament(make_players(4), TournamentConfig(players_per_table=1))


def test_players_reset_to_starting_stack():
    players = make_players(4)

    for player in players:
        player.chips = 5000

    total = len(players) * 1000

    result = run_tournament(players, TournamentConfig(seed=2))

    assert result.winner.chips > 0
    assert result.winner.chips == total  # champion holds the entire pool


def test_print_result_output(capsys):
    result = run_tournament(make_players(4), TournamentConfig(seed=4))

    print_result(result)

    out = capsys.readouterr().out

    assert "Tournament result" in out
    assert "Champion" in out
    assert "Finishing order" in out
    assert result.winner.name in out


def test_result_type():
    result = run_tournament(make_players(4), TournamentConfig(seed=6))

    assert isinstance(result, TournamentResult)
