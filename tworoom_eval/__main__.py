"""Package entry point for tworoom_eval."""

from __future__ import annotations

import argparse

from .eval_cf import run as run_counterfactual
from .eval_factual import run as run_factual


def main() -> None:
    parser = argparse.ArgumentParser(description="TwoRoom evaluation package entry point")
    parser.add_argument("mode", choices=["factual", "counterfactual"])
    args, _ = parser.parse_known_args()

    if args.mode == "factual":
        run_factual()
    else:
        run_counterfactual()


if __name__ == "__main__":
    main()