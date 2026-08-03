from poker.card import Card
from poker.evaluator import evaluate, hand_name

def cards(values, suit="spades"):
    return [Card(rank, suit) for rank in values]

def test_straight_flush():
    score = evaluate(cards([10, 11, 12, 13, 14]))
    assert score[0] == 8
    assert hand_name(score) == "Straight Flush"

def test_pair():
    hand = [
        Card(14, "spades"), Card(14, "hearts"),
        Card(10, "clubs"), Card(7, "diamonds"), Card(2, "clubs")
    ]
    assert evaluate(hand)[0] == 1
