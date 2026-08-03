from poker.card import Card

def test_card_string():
    assert str(Card(14, "spades")) == "AS"
