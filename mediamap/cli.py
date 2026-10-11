from __future__ import annotations

import argparse
import sys

from . import books, music


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="mediamap",
        usage="%(prog)s {books,cds,lps} <command> [options]",
        description="Manage separate semantic maps for books, CDs, and LPs.",
    )
    if not argv or argv[0] in {"-h", "--help"}:
        parser.print_help()
        return
    if argv[0] not in {"books", "cds", "lps"}:
        parser.error(f"unknown collection: {argv[0]}")

    collection, *command_args = argv
    if collection == "books":
        books.main(command_args)
    else:
        music.main(collection, command_args)
