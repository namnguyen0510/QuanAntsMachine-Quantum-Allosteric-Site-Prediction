"""
Command-line interface for QuanAllo.

Usage
-----
    quanallo predict --apo APO.pdb --ligand GDP --method dqaw_lifetime
    quanallo predict --apo APO.pdb --holo HOLO.pdb --ligand GDP --drug MOV
                     --method adaptive_quanant -o results.csv
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from quanallo import AllostericPredictor, METHOD_REGISTRY, __version__


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quanallo",
        description="Quantum allosteric site prediction",
    )
    p.add_argument("--version", action="version", version=f"quanallo {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    # predict subcommand
    pp = sub.add_parser("predict", help="Predict allosteric residues from PDB")
    pp.add_argument("--apo", required=True, help="Path to APO PDB file")
    pp.add_argument("--holo", help="Path to HOLO PDB file (optional)")
    pp.add_argument(
        "--ligand",
        help="HETATM ligand name to auto-detect active site (e.g. 'GDP' for KRAS).",
    )
    pp.add_argument(
        "--drug",
        help="HETATM drug name in HOLO for ground-truth detection (optional).",
    )
    pp.add_argument(
        "--method",
        default="dqaw_lifetime",
        help="Method name. Choices: " + ", ".join(sorted(METHOD_REGISTRY))
             + ", quanant, adaptive_quanant",
    )
    pp.add_argument("--top-k", type=int, default=5)
    pp.add_argument(
        "--selection",
        default="argmax",
        choices=["argmax", "mmr"],
        help="Top-k selection mode.",
    )
    pp.add_argument(
        "-o", "--output",
        help="Output CSV path (writes top-k hits).",
    )
    pp.add_argument("--quiet", action="store_true")

    # list-methods
    sub.add_parser("list-methods", help="List all available methods")
    return p


def cmd_predict(args) -> int:
    predictor = AllostericPredictor(
        method=args.method,
        top_k=args.top_k,
        selection=args.selection,
        verbose=not args.quiet,
    )
    if not args.quiet:
        print(f"[quanallo] running '{args.method}' on {args.apo}", file=sys.stderr)
    result = predictor.predict_from_pdb(
        apo_pdb=args.apo,
        holo_pdb=args.holo,
        auto_active_site_ligand=args.ligand,
        holo_drug_name=args.drug,
    )
    df = result.to_dataframe()
    if args.output:
        df.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"[quanallo] wrote {args.output}", file=sys.stderr)
    else:
        print(df.to_string(index=False))
    if result.weighted_top5 is not None and not args.quiet:
        print(f"\nWeighted top-{args.top_k}: {result.weighted_top5:.3f}",
              file=sys.stderr)
    return 0


def cmd_list_methods(_args) -> int:
    for name, cls in sorted(METHOD_REGISTRY.items()):
        inst = cls()
        print(f"  {name:<16} kind={inst.kind:<18} {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else ''}")
    print("\nMeta methods:")
    print("  quanant            kind=quanant            ant colony of perturbed methods")
    print("  adaptive_quanant   kind=quanant            APO→HOLO transfer with online weights")
    return 0


def main(argv=None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.cmd == "predict":
        return cmd_predict(args)
    if args.cmd == "list-methods":
        return cmd_list_methods(args)
    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
