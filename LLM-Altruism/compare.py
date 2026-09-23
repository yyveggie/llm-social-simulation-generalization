from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent
PAPER_CSV = (
    REPO
    / "johnson_obradovich_NHB_replication_files_04_07_2025"
    / "nature_2023-01-00290_rep_files"
    / "red2dg_collected_coded_01_01_2023.csv"
)


ENGEL_BINS = np.arange(0, 1.05, 0.1)
ENGEL_FREQ = np.array(
    [0.3611, 0.0914, 0.0881, 0.0891, 0.0723, 0.1674, 0.0389, 0.0197, 0.0107, 0.0070, 0.0544]
)

NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _round_to(values: np.ndarray, step: float) -> np.ndarray:
    return np.round(values / step) * step


def _paper_davinci_distributions() -> tuple[np.ndarray, np.ndarray]:
    if not PAPER_CSV.exists():
        raise FileNotFoundError(f"Original paper CSV not found at {PAPER_CSV}")
    df = pd.read_csv(PAPER_CSV, encoding="latin-1")

    df = df[df["unclear5"].isna()].copy()

    def _first_int(s):
        if pd.isna(s):
            return None
        m = NUMBER_RE.search(str(s))
        return float(m.group()) if m else None

    df["resp_n"] = df["respon"].map(_first_int)

    if "interp_share" in df.columns:
        df.loc[df["interp_share"].notna(), "resp_n"] = df.loc[
            df["interp_share"].notna(), "interp_share"
        ]
    df = df.dropna(subset=["resp_n"])
    df["propshare"] = df["resp_n"] / df["stakes"]
    df = df[(df["propshare"] >= 0) & (df["propshare"] <= 1)]

    human = df.loc[df["condit"] == "davinci_experimenter", "propshare"].to_numpy()
    ai = df.loc[
        df["condit"].isin(["davinci_ada", "davinci_babbage", "davinci_curie"]),
        "propshare",
    ].to_numpy()
    return human, ai


def _modern_runs(filter_models: list[str] | None) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    root = REPO / "results"
    if root.exists():
        for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            if filter_models and model_dir.name not in filter_models:
                continue
            base = model_dir / "baseline_dictator"
            if not base.is_dir():
                continue
            cands = [d / "parsed.csv" for d in base.iterdir() if d.is_dir() and (d / "parsed.csv").is_file()]
            if cands:
                runs[model_dir.name] = max(cands, key=lambda p: p.stat().st_mtime)
    legacy = REPO / "results" / "baseline_dictator"
    if legacy.exists():
        for d in sorted(p for p in legacy.iterdir() if p.is_dir()):
            parsed = d / "parsed.csv"
            if not parsed.exists() or d.name in runs:
                continue
            if filter_models and d.name not in filter_models:
                continue
            runs[d.name] = parsed
    return runs


def _modern_distributions(parsed_csv: Path) -> dict[str, np.ndarray]:
    df = pd.read_csv(parsed_csv)
    df = df.dropna(subset=["propshare"])
    df = df[(df["propshare"] >= 0) & (df["propshare"] <= 1)]
    out: dict[str, np.ndarray] = {}
    out["human"] = df.loc[df["condit"] == "experimenter", "propshare"].to_numpy()
    out["charity"] = df.loc[df["condit"] == "charity", "propshare"].to_numpy()
    out["ai"] = df.loc[
        df["condit"].isin(["other_llm_ada", "other_llm_babbage", "other_llm_curie"]),
        "propshare",
    ].to_numpy()
    return out


def _hist_freq(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros_like(ENGEL_BINS)
    rounded = _round_to(values, 0.1).round(2)
    out = np.zeros_like(ENGEL_BINS)
    for i, b in enumerate(ENGEL_BINS):
        out[i] = float(np.isclose(rounded, round(b, 1)).sum())
    total = out.sum()
    if total > 0:
        out /= total
    return out


def _ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(values) == 0:
        return np.array([0, 1]), np.array([0, 0])
    xs = np.sort(values)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    return xs, ys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-m", "--model", action="append",
        help="Only include this modern model (matches subdir name "
             "under results/baseline_dictator/). Repeatable.",
    )
    parser.add_argument(
        "-o", "--out",
        default=str(REPO / "results" / "baseline_dictator" / "comparison.png"),
        help="Output image path.",
    )
    args = parser.parse_args()

    modern = _modern_runs(args.model)
    if not modern:
        print(
            "No parsed.csv files found under results/baseline_dictator/. "
            "Run `python run.py -e baseline_dictator` first.",
            file=sys.stderr,
        )
        return 2

    print("Reconstructing text-davinci-003 distribution from paper CSV...")
    paper_h, paper_ai = _paper_davinci_distributions()
    print(f"  paper davinci-to-human n={len(paper_h)}, davinci-to-AI n={len(paper_ai)}")

    rows = 2
    cols = 1 + len(modern)
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 6.4), squeeze=False)


    ax = axes[0][0]
    ax.bar(ENGEL_BINS - 0.015, ENGEL_FREQ, width=0.03, color="#888", label="Engel 2011 (humans)")
    ax.bar(ENGEL_BINS + 0.015, _hist_freq(paper_h), width=0.03, color="#2a8", label="paper · davinci→human")
    ax.set_title("Reference: humans vs. text-davinci-003")
    ax.set_xlabel("proportion shared")
    ax.set_ylabel("relative frequency")
    ax.legend(fontsize=8)

    ax = axes[1][0]
    for label, arr, color in (
        ("Engel 2011 (humans)", None, "#888"),
        ("paper · davinci→human", paper_h, "#2a8"),
        ("paper · davinci→AI",   paper_ai, "#28a"),
    ):
        if arr is None:
            xs = ENGEL_BINS
            ys = np.cumsum(ENGEL_FREQ)
            ax.plot(xs, ys, color=color, lw=2, label=label)
        else:
            xs, ys = _ecdf(arr)
            ax.plot(xs, ys, color=color, lw=1.5, label=label)
    ax.set_title("ECDF — reference")
    ax.set_xlabel("proportion shared")
    ax.set_ylabel("cumulative frequency")
    ax.legend(fontsize=8, loc="lower right")


    for k, (name, parsed) in enumerate(modern.items(), start=1):
        d = _modern_distributions(parsed)
        ax = axes[0][k]
        ax.bar(ENGEL_BINS - 0.015, _hist_freq(d["human"]), width=0.03, color="#a82", label=f"{name}→human (n={len(d['human'])})")
        ax.bar(ENGEL_BINS + 0.015, _hist_freq(d["ai"]),    width=0.03, color="#28a", label=f"{name}→AI (n={len(d['ai'])})")
        ax.set_title(f"Modern: {name}")
        ax.set_xlabel("proportion shared")
        ax.legend(fontsize=8)

        ax = axes[1][k]
        ax.plot(ENGEL_BINS, np.cumsum(ENGEL_FREQ), color="#888", lw=2, label="Engel 2011 (humans)")
        for cond_name, color in (("human", "#a82"), ("ai", "#28a"), ("charity", "#852")):
            xs, ys = _ecdf(d[cond_name])
            ax.plot(xs, ys, color=color, lw=1.5, label=f"{name}→{cond_name}")
        ax.set_title(f"ECDF — {name}")
        ax.set_xlabel("proportion shared")
        ax.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
