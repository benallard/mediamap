from __future__ import annotations

import argparse

from .books import BookSpec
from .collection import MediaSpec, add_commands, run_command
from .music import MusicSpec

# One line per collection: adding a media type means adding a spec here.
COLLECTIONS: dict[str, MediaSpec] = {
    "books": BookSpec(),
    "cds": MusicSpec("cds"),
    "lps": MusicSpec("lps"),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mediamap",
        description="Manage separate semantic maps for books, CDs, and LPs.",
    )
    collections = parser.add_subparsers(dest="collection")
    for name, spec in COLLECTIONS.items():
        sub = collections.add_parser(
            name,
            help=f"manage the {spec.label} collection",
            description=f"Manage the {spec.label} collection with a frozen {spec.dims}-D semantic map.",
        )
        add_commands(sub, spec)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.collection is None:
        parser.print_help()
        return
    run_command(COLLECTIONS[args.collection], args)