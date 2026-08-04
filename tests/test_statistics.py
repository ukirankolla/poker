from poker.statistics import OpponentStats, StatisticsTracker


def _two_hand_sequence():
    tracker = StatisticsTracker()

    # Hand one: A raises, B calls, both reach showdown.
    tracker.record("A", "raise", "preflop", 0, 0)
    tracker.record("B", "call", "preflop", 1, 20)
    tracker.end_hand(True, {"A", "B"}, {"A", "B"})

    # Hand two: B re-raises (3-bet) and A folds to it.
    tracker.record("A", "fold", "preflop", 2, 50)
    tracker.record("B", "raise", "preflop", 1, 50)
    tracker.record("B", "bet", "flop", 0, 0)
    tracker.end_hand(False, set(), {"A", "B"})

    return tracker


def test_vpip_and_pfr_tracked():
    tracker = _two_hand_sequence()
    stats = tracker.snapshot()["A"]

    assert stats.hands == 2
    assert stats.vpip == 0.5
    assert stats.pfr == 0.5


def test_three_bet_and_fold_to_three_bet():
    tracker = _two_hand_sequence()
    a = tracker.snapshot()["A"]
    b = tracker.snapshot()["B"]

    assert a.fold_to_three_bet == 1.0
    assert b.three_bet == 0.5


def test_aggression_factor():
    tracker = _two_hand_sequence()
    b = tracker.snapshot()["B"]

    # B raised preflop (the 3-bet) and bet the flop = 2 aggressive
    # actions, and called once in hand one.
    assert b.aggression == 2.0


def test_showdown_rate():
    tracker = _two_hand_sequence()
    a = tracker.snapshot()["A"]
    b = tracker.snapshot()["B"]

    assert a.showdown == 0.5
    assert b.showdown == 0.5


def test_snapshot_excludes_one_player():
    tracker = _two_hand_sequence()

    snapshot = tracker.snapshot(exclude="A")

    assert "A" not in snapshot
    assert "B" in snapshot


def test_opponent_stats_aggregate_after_each_hand():
    tracker = StatisticsTracker()

    tracker.record("A", "check", "preflop", 0, 0)
    tracker.end_hand(False, set(), {"A", "B"})

    stats = tracker.snapshot()["A"]
    assert stats.hands == 1
    assert stats.vpip == 0.0


def test_default_stats_are_zero():
    stats = OpponentStats("X")

    assert stats.hands == 0
    assert stats.vpip == 0.0
    assert stats.pfr == 0.0
    assert stats.three_bet == 0.0
    assert stats.aggression == 0.0
    assert stats.showdown == 0.0


def test_save_and_load_round_trip(tmp_path):
    tracker = _two_hand_sequence()

    path = tmp_path / "profile.json"
    tracker.save(path)

    restored = StatisticsTracker.load(path)
    assert restored.to_dict() == tracker.to_dict()

    a = restored.snapshot()["A"]
    b = restored.snapshot()["B"]

    assert a.hands == 2
    assert a.vpip == 0.5
    assert a.fold_to_three_bet == 1.0
    assert b.three_bet == 0.5


def test_from_dict_rebuilds_tracker():
    tracker = _two_hand_sequence()

    restored = StatisticsTracker.from_dict(tracker.to_dict())

    assert set(restored.snapshot()) == {"A", "B"}
    assert restored.snapshot()["B"].aggression == 2.0


def test_from_dict_accepts_empty():
    restored = StatisticsTracker.from_dict({})

    assert restored.snapshot() == {}


def test_from_dict_ignores_missing_file_data():
    restored = StatisticsTracker.from_dict(None)

    assert restored.snapshot() == {}
