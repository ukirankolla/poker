"""Monte Carlo equity estimation for Texas Hold'em."""

import random

from .card import Card, SUITS, RANKS
from .evaluator import evaluate


def _deck_without(excluded_cards):
    excluded = set(excluded_cards)
    deck = []

    for suit in SUITS:
        for rank in RANKS:
            card = Card(rank, suit)
            if card not in excluded:
                deck.append(card)

    return deck


def estimate_equity(
    hole_cards,
    community_cards=(),
    num_opponents=1,
    trials=1000,
    seed=None,
):
    """Estimate the probability that ``hole_cards`` wins (or ties) the pot.

    Every trial deals the remaining board cards and ``num_opponents``
    random two-card hands from the unseen deck, then compares the best
    five-card hands. Split pots award each tied player an equal share
    of the trial, so the result is the share of the pot the hand can
    expect to win against that many random opponents.
    """
    if trials < 1:
        raise ValueError("trials must be at least 1")

    if num_opponents <= 0:
        return 1.0

    hole = tuple(hole_cards)
    community = tuple(community_cards)

    if len(community) > 5:
        raise ValueError("community_cards cannot exceed five cards")

    deck = _deck_without(hole + community)
    board_needed = 5 - len(community)
    cards_needed = board_needed + 2 * num_opponents

    if cards_needed > len(deck):
        raise ValueError("not enough cards for the requested rollouts")

    rng = random.Random(seed)
    won = 0.0

    for _ in range(trials):
        deal = rng.sample(deck, cards_needed)
        board = community + tuple(deal[:board_needed])

        our_score = evaluate(hole + board)

        best_opponent = None
        tied_opponents = 0

        for index in range(num_opponents):
            start = board_needed + 2 * index
            opponent_score = evaluate(
                (deal[start], deal[start + 1]) + board
            )

            if best_opponent is None or opponent_score > best_opponent:
                best_opponent = opponent_score
                tied_opponents = 1
            elif opponent_score == best_opponent:
                tied_opponents += 1

        if our_score > best_opponent:
            won += 1.0
        elif our_score == best_opponent:
            won += 1.0 / (tied_opponents + 1)

    return won / trials


def pot_odds(to_call, pot):
    """Pot odds as a fraction: ``to_call / (pot + to_call)``."""
    if to_call <= 0:
        return 0.0

    return to_call / (pot + to_call)
