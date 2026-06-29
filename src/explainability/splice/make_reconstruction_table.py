"""Format eval_reconstruction.py JSON outputs into the paper's LaTeX table.

Expects one JSON file per (model, dataset) pair in --outputs_dir, named
"{model_key}__{dataset_key}.json" (the convention used by
scripts/run_eval_reconstruction.sh), e.g. "geoclip__dense_grid_val.json".
"""

import argparse
from pathlib import Path
import json

MODELS = [
    ("geoclip", "GeoCLIP"),
    ("satclip", "SatCLIP"),
    ("climplicit", "Climplicit"),
    ("csp_fmow", "CSP-fMoW"),
]

DATASETS = [
    ("dense_grid_val", "UAR"),
    ("geoyfcc_subset_val", "Human-visited"),
]

HEADER = r"""\setlength{\tabcolsep}{4pt}
\resizebox{\linewidth}{!}{%
\begin{tabular}{l ccc ccc}
\toprule
 & \multicolumn{3}{c}{\textbf{UAR}} & \multicolumn{3}{c}{\textbf{Human-visited}} \\
\cmidrule(lr){2-4} \cmidrule(lr){5-7}
\textbf{Model} &
\textbf{MSE} $\downarrow$ &
\textbf{Cos. Sim.} $\uparrow$ &
\textbf{\# Concepts} &
\textbf{MSE} $\downarrow$ &
\textbf{Cos. Sim.} $\uparrow$ &
\textbf{\# Concepts} \\
\midrule"""

FOOTER = r"""\bottomrule
\end{tabular}
}"""


def load_report(outputs_dir, model_key, dataset_key, suffix):
    path = outputs_dir / f"{model_key}__{dataset_key}{suffix}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing eval_reconstruction.py output: {path}")
    return json.loads(path.read_text())


def format_metric(mean, std, include_std, decimals=3):
    cell = f"{mean:.{decimals}f}"
    if include_std:
        cell += rf" $\pm$ {std:.{decimals}f}"
    return cell


def build_row(outputs_dir, model_key, model_name, suffix, include_std):
    reports = {
        dataset_key: load_report(outputs_dir, model_key, dataset_key, suffix)
        for dataset_key, _ in DATASETS
    }

    lambdas = {report["l1_penalty"] for report in reports.values()}
    if len(lambdas) > 1:
        raise ValueError(
            f"{model_name}: l1_penalty differs across datasets: {lambdas}"
        )

    cells = [model_name]
    for dataset_key, _ in DATASETS:
        report = reports[dataset_key]
        cells.append(
            format_metric(report["mse_mean"], report["mse_std"], include_std)
        )
        cells.append(
            format_metric(
                report["cosine_sim_mean"], report["cosine_sim_std"], include_std
            )
        )
        cells.append(
            format_metric(
                report["active_concepts_mean"],
                report["active_concepts_std"],
                include_std,
                decimals=1,
            )
        )
    return cells


def build_table(outputs_dir, suffix, include_std):
    rows = [
        build_row(outputs_dir, model_key, model_name, suffix, include_std)
        for model_key, model_name in MODELS
    ]
    body = "\n".join(" & ".join(cells) + r" \\" for cells in rows)
    return "\n".join([HEADER, body, FOOTER])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs_dir",
        default=str(Path(__file__).parent / "outputs"),
        help="Directory containing eval_reconstruction.py JSON output files",
    )
    parser.add_argument(
        "--small_subset",
        action="store_true",
        help="Use the *_small_subset.json results instead of the full concept dictionary",
    )
    parser.add_argument(
        "--include_std",
        action="store_true",
        help="Append standard deviation (mean $\\pm$ std) to each metric",
    )
    parser.add_argument("--output", help="Optional path to write the LaTeX table to")
    args = parser.parse_args()

    suffix = "_small_subset" if args.small_subset else ""
    table = build_table(Path(args.outputs_dir), suffix, args.include_std)

    print(table)
    if args.output:
        Path(args.output).write_text(table + "\n")


if __name__ == "__main__":
    main()
