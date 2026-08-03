from collections import Counter
from itertools import combinations

HAND_NAMES = {
    8: "Straight Flush",
    7: "Four of a Kind",
    6: "Full House",
    5: "Flush",
    4: "Straight",
    3: "Three of a Kind",
    2: "Two Pair",
    1: "Pair",
    0: "High Card",
}

def _straight_high(ranks):
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    for i in range(len(unique) - 4):
        window = unique[i:i+5]
        if window[0] - window[4] == 4:
            return window[0]
    return None

def evaluate_five(cards):
    ranks = [c.rank for c in cards]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    flush = len({c.suit for c in cards}) == 1
    straight = _straight_high(ranks)

    if flush and straight:
        return (8, straight)
    quads = sorted((r for r, c in counts.items() if c == 4), reverse=True)
    if quads:
        q = quads[0]
        return (7, q, max(r for r in ranks if r != q))
    trips = sorted((r for r, c in counts.items() if c == 3), reverse=True)
    pairs = sorted((r for r, c in counts.items() if c == 2), reverse=True)
    if trips and pairs:
        return (6, trips[0], pairs[0])
    if flush:
        return (5, *sorted(ranks, reverse=True))
    if straight:
        return (4, straight)
    if trips:
        kickers = sorted((r for r in ranks if r != trips[0]), reverse=True)
        return (3, trips[0], *kickers)
    if len(pairs) >= 2:
        high, low = pairs[:2]
        kicker = max(r for r in ranks if r not in (high, low))
        return (2, high, low, kicker)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = sorted((r for r in ranks if r != pair), reverse=True)
        return (1, pair, *kickers)
    return (0, *sorted(ranks, reverse=True))

def evaluate(cards):
    if len(cards) < 5:
        raise ValueError("at least five cards are required")
    return max(evaluate_five(combo) for combo in combinations(cards, 5))

def hand_name(score):
    return HAND_NAMES[score[0]]
