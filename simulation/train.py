"""Learn a poker action policy from self-play data.

``simulation.self_play`` collects one JSON record per decision. This
module reads those records and fits a multinomial logistic regression
(softmax) policy with plain-Python batch gradient descent, then exports
the trained policy so it can be loaded by ``LearnedPolicyAgent`` and
played against the other agents.

The feature vector is a compact encoding of the decision context:

* one-hot street and position
* log-scaled pot, to-call, stack, current bet, minimum raise
* players remaining
* pot odds and the to-call / stack ratio
* average opponent VPIP, PFR, and aggression

Training is fully deterministic (no randomness), which makes the same
data always yield the same policy.
"""

import argparse
import json
import math
import random


# ----------------------------------------------------------------------
# feature extraction
# ----------------------------------------------------------------------

STREETS = ("preflop", "flop", "turn", "river")
POSITIONS = ("button", "small_blind", "big_blind", "middle")

FEATURE_NAMES = (
    *(f"street_{street}" for street in STREETS),
    *(f"position_{position}" for position in POSITIONS),
    "log_pot",
    "log_to_call",
    "log_chips",
    "log_current_bet",
    "log_minimum_raise",
    "players_remaining",
    "pot_odds",
    "stack_ratio",
    "opp_avg_vpip",
    "opp_avg_pfr",
    "opp_avg_aggression",
)


def _onehot(value, options):
    return [1.0 if value == option else 0.0 for option in options]


def _state_from_record(record):
    return {
        "street": record.get("street", "preflop"),
        "position": record.get("position", "middle"),
        "pot": record.get("pot", 0),
        "to_call": record.get("to_call", 0),
        "chips": record.get("chips", 0),
        "current_bet": record.get("current_bet", 0),
        "minimum_raise": record.get("minimum_raise", 0),
        "players_remaining": record.get("players_remaining", 2),
        "opponent_stats": record.get("opponent_stats", {}),
    }


def _state_from_context(context):
    community = len(context.community_cards)
    street = (
        "preflop"
        if community == 0
        else "flop"
        if community == 3
        else "turn"
        if community == 4
        else "river"
    )

    return {
        "street": street,
        "position": context.position,
        "pot": context.pot,
        "to_call": max(0, context.current_bet - context.player_bet),
        "chips": context.chips,
        "current_bet": context.current_bet,
        "minimum_raise": context.minimum_raise,
        "players_remaining": context.players_remaining,
        "opponent_stats": context.opponent_stats,
    }


def _opponent_avg(state):
    vpip = []
    pfr = []
    aggression = []

    for stats in state["opponent_stats"].values():
        if isinstance(stats, dict):
            vpip.append(stats.get("vpip", 0.0))
            pfr.append(stats.get("pfr", 0.0))
            aggression.append(stats.get("aggression", 0.0))
        else:
            vpip.append(getattr(stats, "vpip", 0.0))
            pfr.append(getattr(stats, "pfr", 0.0))
            aggression.append(getattr(stats, "aggression", 0.0))

    def mean(values):
        return sum(values) / len(values) if values else 0.0

    return mean(vpip), mean(pfr), mean(aggression)


def extract_features(state):
    """Turn a decision state into a numeric feature vector."""
    pot = max(0, state["pot"])
    to_call = max(0, state["to_call"])
    chips = max(0, state["chips"])
    current_bet = max(0, state["current_bet"])
    minimum_raise = max(0, state["minimum_raise"])

    pot_odds = to_call / (pot + to_call) if pot + to_call > 0 else 0.0
    stack_ratio = to_call / chips if chips > 0 else 0.0

    avg_vpip, avg_pfr, avg_aggression = _opponent_avg(state)

    return [
        *_onehot(state["street"], STREETS),
        *_onehot(state["position"], POSITIONS),
        math.log1p(pot),
        math.log1p(to_call),
        math.log1p(chips),
        math.log1p(current_bet),
        math.log1p(minimum_raise),
        float(state["players_remaining"]),
        pot_odds,
        stack_ratio,
        avg_vpip,
        avg_pfr,
        avg_aggression,
    ]


def features_from_record(record):
    return extract_features(_state_from_record(record))


def features_from_context(context):
    return extract_features(_state_from_context(context))


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------


def _softmax(logits):
    max_logit = max(logits)
    exponentials = [math.exp(logit - max_logit) for logit in logits]
    total = sum(exponentials)
    return [value / total for value in exponentials]


class Policy:
    """A multinomial logistic regression policy over poker actions."""

    def __init__(self, classes=(), feature_names=(), mean=(), std=()):
        self.classes = list(classes)
        self.feature_names = list(feature_names)
        self.mean = list(mean)
        self.std = list(std)
        self.weights = [
            [0.0] * len(feature_names) for _ in classes
        ]
        self.bias = [0.0] * len(classes)

    def fit(self, records, epochs=30, learning_rate=0.1):
        """Fit on a list of self-play decision records."""
        if not records:
            raise ValueError("at least one record is required to fit")

        self.classes = sorted({record["action"] for record in records})
        self.feature_names = list(FEATURE_NAMES)

        matrix = [features_from_record(record) for record in records]
        targets = [record["action"] for record in records]

        n_features = len(FEATURE_NAMES)

        # Standardize features from the training mean/std.
        means = []
        stds = []

        for index in range(n_features):
            column = [row[index] for row in matrix]
            mean = sum(column) / len(column)
            variance = sum((value - mean) ** 2 for value in column) / len(column)
            means.append(mean)
            stds.append(math.sqrt(variance) if variance > 0 else 1.0)

        self.mean = means
        self.std = stds
        self.weights = [
            [0.0] * n_features for _ in self.classes
        ]
        self.bias = [0.0] * len(self.classes)

        normalized = [
            [
                (row[index] - self.mean[index]) / self.std[index]
                for index in range(n_features)
            ]
            for row in matrix
        ]

        class_index = {action: index for index, action in enumerate(self.classes)}
        labels = [class_index[target] for target in targets]

        for _ in range(epochs):
            gradients = [
                [0.0] * n_features for _ in self.classes
            ]
            gradient_bias = [0.0] * len(self.classes)

            for row, label in zip(normalized, labels):
                probabilities = _softmax(
                    [
                        self.bias[k]
                        + sum(
                            self.weights[k][i] * row[i]
                            for i in range(n_features)
                        )
                        for k in range(len(self.classes))
                    ]
                )

                for k in range(len(self.classes)):
                    error = (1.0 if k == label else 0.0) - probabilities[k]
                    gradient_bias[k] += error

                    for i in range(n_features):
                        gradients[k][i] += error * row[i]

            scale = learning_rate / len(normalized)

            for k in range(len(self.classes)):
                self.bias[k] += scale * gradient_bias[k]

                for i in range(n_features):
                    self.weights[k][i] += scale * gradients[k][i]

        return self

    def _logits(self, features):
        x = [
            (features[i] - self.mean[i]) / self.std[i]
            for i in range(len(features))
        ]

        return [
            self.bias[k]
            + sum(self.weights[k][i] * x[i] for i in range(len(x)))
            for k in range(len(self.classes))
        ]

    def predict_proba(self, features):
        probabilities = _softmax(self._logits(features))
        return {
            action: probability
            for action, probability in zip(self.classes, probabilities)
        }

    def predict(self, features):
        probabilities = self.predict_proba(features)
        return max(probabilities, key=probabilities.get)

    def accuracy(self, records):
        if not records:
            return 0.0

        correct = sum(
            1
            for record in records
            if self.predict(features_from_record(record))
            == record["action"]
        )

        return correct / len(records)

    def to_dict(self):
        return {
            "classes": self.classes,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "std": self.std,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, data):
        policy = cls(
            classes=data["classes"],
            feature_names=data["feature_names"],
            mean=data["mean"],
            std=data["std"],
        )
        policy.weights = data["weights"]
        policy.bias = data["bias"]
        return policy

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# ----------------------------------------------------------------------
# training entry points
# ----------------------------------------------------------------------


def read_records(path):
    records = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def train_policy(records, epochs=30, learning_rate=0.1, test_fraction=0.2):
    """Fit a policy and report train/test accuracy.

    Returns a ``(policy, metrics)`` tuple. The split is deterministic
    when ``records`` is ordered the same way (a seeded shuffle is used).
    """
    if not records:
        raise ValueError("at least one record is required")

    rng = random.Random(42)
    shuffled = list(records)
    rng.shuffle(shuffled)

    split = max(1, int(len(shuffled) * (1 - test_fraction)))
    train, test = shuffled[:split], shuffled[split:]

    policy = Policy().fit(train, epochs=epochs, learning_rate=learning_rate)

    metrics = {
        "train_records": len(train),
        "test_records": len(test),
        "train_accuracy": policy.accuracy(train),
        "test_accuracy": policy.accuracy(test),
    }

    return policy, metrics


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a poker action policy from self-play JSONL."
    )
    parser.add_argument("--data", required=True, help="self-play JSONL file")
    parser.add_argument(
        "--output", default="policy.json", help="output policy JSON file"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    args = parser.parse_args(argv)

    records = read_records(args.data)

    if not records:
        parser.error(f"no records found in {args.data!r}")

    policy, metrics = train_policy(
        records,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        test_fraction=args.test_fraction,
    )

    policy.save(args.output)

    print(f"Trained on {metrics['train_records']:,} records -> {args.output}")
    print(
        f"Accuracy: train {metrics['train_accuracy']:.1%}  "
        f"test {metrics['test_accuracy']:.1%}"
    )
    print(f"Actions learned: {', '.join(policy.classes)}")


if __name__ == "__main__":
    main()
