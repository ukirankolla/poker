from .deck import Deck
from .player import Player
from .evaluator import evaluate, hand_name
from .betting import BettingEngine, BettingPlayer
from .betting_round import BettingRound
from agents.base_agent import DecisionContext


class HoldemGame:
    """Drives one complete Texas Hold'em hand.

    Flow: post blinds -> preflop betting -> flop -> turn -> river ->
    showdown, then award the main pot and any side pots to the best
    eligible hand(s) per pot.

    The dealer button rotates after every hand. Heads-up rules apply
    when exactly two players are seated: the button posts the small
    blind and acts first preflop, while the big blind acts first on
    every later street.

    When a player goes all-in for less than the current bet, the
    matched money forms the main pot and the excess forms one or more
    side pots that only the larger contributors can win.
    """

    def __init__(self, players, seed=None, small_blind=5, big_blind=10):
        if len(players) < 2:
            raise ValueError("at least two players are required")

        self.players = players
        self.seed = seed
        self.small_blind = small_blind
        self.big_blind = big_blind

        self.deck = None
        self.community = []
        self.pot = 0
        self.button_index = -1

        self._betting_players = []
        self._engine = None
        self._hands_played = 0
        self.action_history = []

    # ------------------------------------------------------------------
    # betting engine wiring
    # ------------------------------------------------------------------

    def _create_betting_engine(self):
        self._betting_players = [
            BettingPlayer(name=player.name, stack=player.chips)
            for player in self.players
        ]

        self._engine = BettingEngine(
            self._betting_players,
            minimum_bet=self.big_blind,
        )

        return self._engine

    def _betting_index(self, betting_player):
        """Identity-based lookup so equal-valued BettingPlayers stay distinct."""
        for index, candidate in enumerate(self._betting_players):
            if candidate is betting_player:
                return index

        raise ValueError("player does not belong to this game")

    def _sync_to_players(self):
        for player, betting_player in zip(
            self.players, self._betting_players
        ):
            player.chips = betting_player.stack
            player.folded = betting_player.folded
            player.current_bet = betting_player.contribution
            player.all_in = betting_player.all_in

    # ------------------------------------------------------------------
    # blinds and positions
    # ------------------------------------------------------------------

    def _blind_positions(self):
        """Return the (small_blind, big_blind) player indices."""
        count = len(self.players)

        if count == 2:
            return self.button_index, (self.button_index + 1) % count

        return (
            (self.button_index + 1) % count,
            (self.button_index + 2) % count,
        )

    def _post_blinds(self):
        sb_index, bb_index = self._blind_positions()

        self._engine.state.add_to_pot(
            self._betting_players[sb_index], self.small_blind
        )
        self._engine.state.add_to_pot(
            self._betting_players[bb_index], self.big_blind
        )

        self._engine.state.current_bet = self._betting_players[
            bb_index
        ].contribution

    def _preflop_first_actor(self):
        count = len(self.players)

        if count == 2:
            return self.button_index

        return (self.button_index + 3) % count

    def _postflop_first_actor(self):
        return (self.button_index + 1) % len(self.players)

    def _position_label(self, index):
        count = len(self.players)

        if count == 2:
            if (index - self.button_index) % count == 0:
                return "button"
            return "big_blind"

        distance = (index - self.button_index) % count

        if distance == 0:
            return "button"
        if distance == 1:
            return "small_blind"
        if distance == 2:
            return "big_blind"

        return "middle"

    # ------------------------------------------------------------------
    # agent interaction
    # ------------------------------------------------------------------

    def _legal_actions(self, betting_player):
        state = self._engine.state
        to_call = state.to_call(betting_player)
        stack = betting_player.stack
        minimum = state.minimum_raise

        actions = ["fold"]

        if to_call == 0:
            actions.append("check")

            if stack >= minimum:
                if state.current_bet == 0:
                    actions.append("bet")
                else:
                    actions.append("raise")
        else:
            actions.append("call")

            if stack > to_call and stack >= to_call + minimum:
                actions.append("raise")

        if stack > 0:
            actions.append("all_in")

        return tuple(actions)

    def _build_context(self, player, betting_player, actions):
        index = self._betting_index(betting_player)

        return DecisionContext(
            hole_cards=tuple(player.hole_cards),
            community_cards=tuple(self.community),
            pot=self._engine.state.pot,
            chips=betting_player.stack,
            current_bet=self._engine.state.current_bet,
            player_bet=betting_player.contribution,
            minimum_raise=self._engine.state.minimum_raise,
            position=self._position_label(index),
            players_remaining=sum(
                1 for bp in self._betting_players if not bp.folded
            ),
            allowed_actions=actions,
        )

    def _apply_action(self, betting_round, betting_player, action):
        if action == "fold":
            betting_round.fold()
            return None

        if action == "check":
            betting_round.check()
            return None

        if action == "call":
            return betting_round.call()

        if action == "bet":
            return betting_round.bet(
                self._engine.state.minimum_raise
            )

        if action == "raise":
            to_call = self._engine.state.to_call(betting_player)
            return betting_round.raise_bet(
                to_call + self._engine.state.minimum_raise
            )

        if action == "all_in":
            return betting_round.all_in()

        raise ValueError(f"unknown action: {action!r}")

    def _run_betting_round(self, start_index, street):
        betting_round = BettingRound(
            self._engine, self._betting_players, start_index
        )

        while True:
            betting_player = betting_round.current_player()

            if betting_player is None:
                break

            index = self._betting_index(betting_player)
            player = self.players[index]
            actions = self._legal_actions(betting_player)
            context = self._build_context(player, betting_player, actions)

            action = player.agent.decide(context)

            if action not in actions:
                raise ValueError(
                    f"{player.name} returned illegal action {action!r}; "
                    f"allowed: {actions}"
                )

            amount = self._apply_action(
                betting_round, betting_player, action
            )

            self.action_history.append(
                {
                    "street": street,
                    "player": player.name,
                    "action": action,
                    "amount": amount,
                }
            )

            self._sync_to_players()

        return betting_round

    # ------------------------------------------------------------------
    # hand flow
    # ------------------------------------------------------------------

    def _deal_community(self, count):
        self.community.extend(self.deck.draw_many(count))

    def _single_active_player(self):
        active = [
            bp for bp in self._betting_players if not bp.folded
        ]

        if not active:
            raise ValueError("no active players remain")

        if len(active) == 1:
            return active[0]

        return None

    def _award_slice(self, amount, winners):
        share, remainder = divmod(amount, len(winners))

        for position, (player, betting_player) in enumerate(winners):
            player.chips += share + (1 if position < remainder else 0)
            betting_player.stack = player.chips

    def _award_pot(self, winners):
        pot = self._engine.state.pot
        self._award_slice(pot, winners)
        self.pot = pot

    def _showdown(self, verbose=False):
        main_winners = None
        best = None

        for amount, eligible in self._engine.state.compute_pots():
            if not eligible:
                continue

            results = []

            for betting_player in eligible:
                index = self._betting_index(betting_player)
                player = self.players[index]
                score = evaluate(player.hole_cards + self.community)
                results.append((score, player, betting_player))

            slice_best = max(score for score, _, _ in results)
            winners = [
                (player, betting_player)
                for score, player, betting_player in results
                if score == slice_best
            ]

            self._award_slice(amount, winners)

            if main_winners is None:
                main_winners = [player for player, _ in winners]
                best = slice_best

        if main_winners is None:
            # No money was committed to the pot (every player was busted
            # or checked through with a zero blind). The best remaining
            # hand is still the winner, of an empty pot.
            results = []

            for index, betting_player in enumerate(self._betting_players):
                if betting_player.folded:
                    continue
                player = self.players[index]
                score = evaluate(player.hole_cards + self.community)
                results.append((score, player))

            best = max(score for score, _ in results)
            main_winners = [
                player for score, player in results if score == best
            ]

        self.pot = self._engine.state.pot

        if verbose:
            self._print_summary(main_winners, best=best)

        return main_winners, best

    def _finish_without_showdown(self, betting_player, verbose=False):
        index = self._betting_index(betting_player)
        player = self.players[index]

        self._award_pot([(player, betting_player)])

        if verbose:
            self._print_summary([player], best=None)

        # Category 9 is the "everyone else folded" marker.
        return [player], (9,)

    def _print_summary(self, winners, best=None):
        print(f"Board: {' '.join(map(str, self.community))}")

        for player in self.players:
            print(
                f"{player.name}: "
                f"{' '.join(map(str, player.hole_cards))}"
            )

        if best is None:
            reason = "everyone else folded"
        else:
            reason = hand_name(best)

        print(
            f"Winner: {', '.join(player.name for player in winners)} "
            f"({reason})"
        )

        print(f"Pot: {self.pot}")

    def play_hand(self, verbose=False):
        self.button_index = (self.button_index + 1) % len(self.players)

        self.deck = Deck(
            None
            if self.seed is None
            else self.seed + self._hands_played
        )
        self._hands_played += 1

        self.community = []
        self.pot = 0
        self.action_history = []

        for player in self.players:
            player.reset_for_hand()

        self._create_betting_engine()

        for player in self.players:
            player.hole_cards = self.deck.draw_many(2)

        self._post_blinds()
        self._sync_to_players()

        self._run_betting_round(self._preflop_first_actor(), "preflop")

        sole = self._single_active_player()

        if sole is not None:
            return self._finish_without_showdown(sole, verbose)

        for street, count in (
            ("flop", 3),
            ("turn", 1),
            ("river", 1),
        ):
            self._deal_community(count)
            self._engine.reset_street()
            self._run_betting_round(self._postflop_first_actor(), street)
            self._sync_to_players()

            sole = self._single_active_player()

            if sole is not None:
                return self._finish_without_showdown(sole, verbose)

        return self._showdown(verbose=verbose)
