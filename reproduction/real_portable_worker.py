"""Run fresh real fits/scores with public private-input checksum references."""
from __future__ import annotations
import argparse
from pathlib import Path
import real_worker as real
import real_private_catalog as catalogue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("fit", "score"), required=True)
    for name in ("source-root", "reference-root", "run-root", "catalogue", "exe", "cmdstan"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--scope", choices=("temporal", "territorial"))
    args = parser.parse_args()
    catalogue.install(real, args.catalogue, args.reference_root)
    if args.action == "fit":
        real.fit(args)
    else:
        from real_score import score
        score(args)


if __name__ == "__main__":
    main()
