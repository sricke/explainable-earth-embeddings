#!/usr/bin/env python3
"""
Evaluate generated captions: smoothed sentence BLEU vs references (optional), length stats,
and optional METEOR when NLTK data is available.

Examples::

  python scripts/evaluate_captions.py --preds outputs/captions.csv

  python scripts/evaluate_captions.py --preds preds.csv --refs refs.csv \\
      --ref-col reference --hyp-col caption

  python scripts/evaluate_captions.py --preds preds.csv --reference-in-preds --ref-col reference
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


def _tokenize(s: str) -> List[str]:
    return re.findall(r"\w+", s.lower())


def bleu_one(reference: str, hypothesis: str) -> float:
    ref = [_tokenize(reference)]
    hyp = _tokenize(hypothesis)
    if not hyp or not ref[0]:
        return 0.0
    return float(
        sentence_bleu(
            ref,
            hyp,
            smoothing_function=SmoothingFunction().method1,
        )
    )


def corpus_metrics(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    include_meteor: bool = False,
) -> Dict[str, Any]:
    """Corpus-level BLEU (single reference per line), optional METEOR mean, and length stats."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    bleus = [bleu_one(r, h) for r, h in zip(references, hypotheses)]
    ref_lens = [len(_tokenize(r)) for r in references]
    hyp_lens = [len(_tokenize(h)) for h in hypotheses]
    out: Dict[str, Any] = {
        "n": len(bleus),
        "bleu_mean": float(sum(bleus) / max(len(bleus), 1)),
        "bleu_min": float(min(bleus)) if bleus else 0.0,
        "bleu_max": float(max(bleus)) if bleus else 0.0,
        "ref_tokens_mean": float(sum(ref_lens) / max(len(ref_lens), 1)),
        "hyp_tokens_mean": float(sum(hyp_lens) / max(len(hyp_lens), 1)),
    }
    if include_meteor:
        try:
            import nltk
            from nltk.translate.meteor_score import meteor_score

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            out["meteor_mean"] = float("nan")
        else:
            meteors = []
            for r, h in zip(references, hypotheses):
                rt, ht = _tokenize(r), _tokenize(h)
                if not rt or not ht:
                    meteors.append(0.0)
                else:
                    meteors.append(float(meteor_score(rt, ht)))
            out["meteor_mean"] = float(sum(meteors) / max(len(meteors), 1))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BLEU / length (and optional METEOR) for caption CSVs.",
    )
    parser.add_argument("--preds", required=True, type=Path, help="CSV with hypothesis captions")
    parser.add_argument(
        "--refs",
        type=Path,
        default=None,
        help="CSV with reference captions (same row order as --preds)",
    )
    parser.add_argument(
        "--reference-in-preds",
        action="store_true",
        help="Use --ref-col from --preds instead of a separate --refs file",
    )
    parser.add_argument("--ref-col", default="reference", help="Reference column name")
    parser.add_argument("--hyp-col", default="caption", help="Hypothesis column name in preds")
    parser.add_argument(
        "--meteor",
        action="store_true",
        help="Also report corpus METEOR (downloads NLTK WordNet data on first use)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write metrics as JSON to this path",
    )
    args = parser.parse_args()

    preds_df = pd.read_csv(args.preds)
    if args.hyp_col not in preds_df.columns:
        raise SystemExit(f"Missing hypothesis column {args.hyp_col!r}; columns: {list(preds_df.columns)}")

    hyp = preds_df[args.hyp_col].astype(str).tolist()

    if args.reference_in_preds:
        if args.ref_col not in preds_df.columns:
            raise SystemExit(f"Missing reference column {args.ref_col!r} in preds")
        refs = preds_df[args.ref_col].astype(str).tolist()
        m = corpus_metrics(refs, hyp, include_meteor=args.meteor)
    elif args.refs is not None:
        refs_df = pd.read_csv(args.refs)
        if args.ref_col not in refs_df.columns:
            raise SystemExit(f"Missing column {args.ref_col!r} in refs; got {list(refs_df.columns)}")
        if len(refs_df) != len(preds_df):
            raise SystemExit("refs and preds must have the same number of rows")
        refs = refs_df[args.ref_col].astype(str).tolist()
        m = corpus_metrics(refs, hyp, include_meteor=args.meteor)
    else:
        hyp_lens = [len(_tokenize(h)) for h in hyp]
        m = {
            "n": len(hyp),
            "hyp_tokens_mean": float(sum(hyp_lens) / max(len(hyp_lens), 1)),
            "hyp_tokens_std": float(pd.Series(hyp_lens).std()) if hyp_lens else 0.0,
            "hyp_tokens_min": int(min(hyp_lens)) if hyp_lens else 0,
            "hyp_tokens_max": int(max(hyp_lens)) if hyp_lens else 0,
        }

    for k, v in m.items():
        if isinstance(v, float):
            if v != v:  # nan
                print(f"{k}: nan")
            else:
                print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.output_json is not None:
        # JSON does not like nan; replace with null
        serializable = {k: (None if isinstance(v, float) and v != v else v) for k, v in m.items()}
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(serializable, indent=2))
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
