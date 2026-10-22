"""
Inspect model architecture and module composition.

This script loads one or more serialised model checkpoints and prints
their internal module hierarchy.  An optional keyword filter narrows
the output to modules whose class name contains any of the supplied
terms — useful for locating dropout, normalisation, or attention layers
across different backbone architectures.

Usage
-----
# Full architecture of a single checkpoint
python inspect_model.py --models checkpoint.pt

# Filter to dropout-related layers across several checkpoints
python inspect_model.py --models a.pt b.pt --keywords drop dropout

# Include a parameter/layer summary
python inspect_model.py --models checkpoint.pt --summary -v
"""

from __future__ import annotations

import logging
import warnings
from argparse import (
    ArgumentDefaultsHelpFormatter,
    ArgumentParser,
    Namespace,
)

from utils import load_model

warnings.filterwarnings("ignore")


LOGGER = logging.getLogger(__name__)


def configure_logging(verbose: int = 0, quiet: bool = False) -> None:
    """
    Configure console logging verbosity.

    Parameters
    ----------
    verbose : int, default=0
        Verbosity level obtained by counting ``-v`` flags on the command
        line.  0 → WARNING, 1 → INFO, 2+ → DEBUG.
    quiet : bool, default=False
        When *True*, suppress all messages below ERROR regardless of
        *verbose*.
    """
    if quiet:
        level = logging.ERROR
    elif verbose >= 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def build_parser() -> ArgumentParser:
    """
    Build the argument parser for the model inspection script.

    Returns
    -------
    argparse.ArgumentParser
        Configured command-line parser.
    """
    parser = ArgumentParser(
        description=(
            "Inspect the architecture of one or more model checkpoints. "
            "Prints each module's fully-qualified name and class, "
            "with an optional keyword filter to target specific layer types "
            "(e.g. 'drop', 'norm', 'attn')."
        ),
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="One or more paths to model checkpoints to inspect.",
    )
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional case-insensitive substrings to filter modules by class "
            "name (e.g. --keywords drop norm).  When omitted, all modules are "
            "printed."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity (-v INFO, -vv DEBUG).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all messages below ERROR level.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a parameter/layer summary after the module listing.",
    )
    return parser


def parse_arguments(
    known_args: list[str] | None = None,
    parser: ArgumentParser | None = None,
) -> Namespace:
    """
    Parse and validate command-line arguments.

    Parameters
    ----------
    known_args : list[str] or None, default=None
        Explicit argument list used for programmatic invocation or testing.
        When *None*, ``sys.argv[1:]`` is used.
    parser : ArgumentParser or None, default=None
        Pre-built parser to use.  When *None*, ``build_parser()`` is called.

    Returns
    -------
    argparse.Namespace
        Parsed and validated arguments.

    Raises
    ------
    ValueError
        If ``--models`` is not provided.
    """
    if parser is None:
        parser = build_parser()

    args = parser.parse_args(known_args)

    for required in ("models",):
        if getattr(args, required) is None:
            raise ValueError(f"`{required}` is required.")

    return args


def main(known_args: list[str] | None = None) -> None:
    """
    Entry point for the model inspection script.

    Loads each checkpoint in turn, then either prints all named modules or
    filters them by keyword.  Failures to load a checkpoint are logged and
    skipped so that the remaining models are still inspected.

    Parameters
    ----------
    known_args : list[str] or None, default=None
        Forwarded to :func:`parse_arguments`.  Useful for calling ``main``
        programmatically without touching ``sys.argv``.
    """
    args = parse_arguments(known_args=known_args)

    configure_logging(verbose=args.verbose, quiet=args.quiet)

    for path in args.models:
        LOGGER.info("Inspecting model: %s", path)

        try:
            model = load_model(path)
        except Exception:
            LOGGER.warning("Unable to load model: %s", path)
            continue

        print(f"\n__________ modules __________")

        if args.keywords is not None:
            keywords = {k.lower() for k in args.keywords}
            for name, m in model.network.named_modules():
                if any(k in m.__class__.__name__.lower() for k in keywords):
                    print(
                        f"  {name:60s}  {m.__class__.__name__:30s}  training={m.training}"
                    )
        else:
            for name, m in model.network.named_modules():
                print(f"  {name:60s}  {m.__class__.__name__}")

        if args.summary:
            print(f"\n__________ summary __________")
            model.network.print_summary()


if __name__ == "__main__":
    main()
