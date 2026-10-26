"""Generate data, train three networks, then compare with Newton-FGMRES."""

import argparse
from pathlib import Path

from lpm_gnn.nonlinear.data import generate
from lpm_gnn.nonlinear.evaluation import evaluate
from lpm_gnn.nonlinear.training import METHODS, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    data = commands.add_parser(
        "generate", help="Create all synthetic data and the common linear prior."
    )
    data.add_argument("--output", default="outputs/nonlinear")
    data.add_argument("--training", type=int, default=512)
    data.add_argument("--validation", type=int, default=128)
    data.add_argument("--test", type=int, default=256)
    data.add_argument("--training-seed", type=int, default=20261021)
    data.add_argument("--validation-seed", type=int, default=20261022)
    data.add_argument("--test-seed", type=int, default=20261101)
    training = commands.add_parser("train")
    training.add_argument("--run", type=Path, default=Path("outputs/nonlinear"))
    training.add_argument("--method", choices=["all", *METHODS], default="all")
    training.add_argument("--steps", type=int, default=4000)
    training.add_argument("--batch-size", type=int, default=16)
    training.add_argument("--seed", type=int, default=202610211)
    training.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    training.add_argument("--checkpoint-every", type=int, default=100)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, default=Path("outputs/nonlinear"))
    evaluation.add_argument("--output", default="outputs/nonlinear_evaluation")
    evaluation.add_argument("--repeats", type=int, default=5)
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.command == "generate":
        generate(
            args.output,
            args.training,
            args.validation,
            args.test,
            args.training_seed,
            args.validation_seed,
            args.test_seed,
        )
    elif args.command == "train":
        for method in METHODS if args.method == "all" else (args.method,):
            train(
                args.run,
                method,
                args.steps,
                args.batch_size,
                args.seed,
                args.device,
                args.checkpoint_every,
            )
    else:
        evaluate(args.run, args.output, args.repeats, args.limit, args.plot)


if __name__ == "__main__":
    main()
