from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import schemas as _schemas

logger = logging.getLogger(__name__)


def _parse_row(task_type: str, respon, stakes):
    fn = getattr(_schemas, "parse_for_task", None)
    if fn is not None:
        return fn(task_type, respon, stakes)
    upper = _schemas.upper_for(task_type, stakes)
    return _schemas.parse_response(respon, upper)


_VALIDATION_PER_BUCKET = 12
_VALIDATION_SEED = 8


def analyze_csv(
    raw_csv: Path,
    out_dir: Path,
    *,
    task_type: str,
    experiment_name: str,
    model_name: str,
    payoff_max: dict | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)


    df = pd.read_csv(raw_csv, keep_default_na=False)
    if df.empty:
        logger.warning("No rows in %s", raw_csv)
        return {"trials": 0}

    decisions: list[float | None] = []
    unusables: list[str | None] = []
    paths: list[str | None] = []
    for _, row in df.iterrows():
        try:

            pr = _parse_row(task_type, row.get("respon"), row.get("stakes"))
        except (TypeError, ValueError):
            decisions.append(None)
            unusables.append("bad_stakes")
            paths.append(None)
            continue
        decisions.append(pr.decision)
        unusables.append(pr.unusable_reason)

        paths.append(getattr(pr, "parse_path", None))

    df["decision"] = decisions
    df["unusable_reason"] = unusables
    df["parse_path"] = paths
    if task_type == "percent":
        df["propshare"] = df["decision"] / 100.0
    else:
        df["propshare"] = df["decision"] / df["stakes"]

    parsed_path = out_dir / "parsed.csv"
    df.to_csv(parsed_path, index=False)


    _export_validation_sample(df, out_dir / "validation_sample.csv")


    if task_type in ("nonsocial", "nonsocial_insertion"):
        summary_df = _summarize_nonsocial(df, payoff_max or {})
    else:

        summary_df = _summarize_dictator(df)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)


    fig_path = out_dir / "histograms.png"
    if task_type in ("nonsocial", "nonsocial_insertion"):
        _plot_nonsocial(df, model_name, experiment_name, fig_path)
    else:
        _plot_dictator(df, model_name, experiment_name, fig_path,
                       xlabel=("percent / 100" if task_type == "percent" else "proportion shared"))

    return {
        "trials": int(len(df)),
        "usable": int(df["decision"].notna().sum()),
        "summary_csv": str(summary_path),
        "parsed_csv": str(parsed_path),
        "figure": str(fig_path),
    }


def _export_validation_sample(df: pd.DataFrame, out_path: Path) -> None:
    def bucket(row) -> str:
        reason = row.get("unusable_reason")
        if isinstance(reason, str) and reason:
            if reason == "ambiguous":
                return "ambiguous"
            if reason == "refusal":
                return "refusal"
            if reason in ("api_error", "empty"):
                return ""
            return "other_unusable"
        path = row.get("parse_path")
        if path == "fallback_first":
            return "ok_fallback"
        if path == "insertion_tail":
            return "ok_insertion"
        return "ok_standalone"

    work = df.copy()
    work["_bucket"] = work.apply(bucket, axis=1)
    parts = []
    for name, sub in work[work["_bucket"] != ""].groupby("_bucket"):
        parts.append(sub.sample(n=min(_VALIDATION_PER_BUCKET, len(sub)),
                                random_state=_VALIDATION_SEED))
    if not parts:
        return
    sample = pd.concat(parts)
    cols = [c for c in ("indexx", "condit", "stakes", "respon", "decision",
                        "propshare", "unusable_reason", "parse_path", "_bucket") if c in sample.columns]
    sample = sample[cols].rename(columns={"_bucket": "bucket"})
    sample["human_decision"] = ""
    sample["human_note"] = ""
    sample.to_csv(out_path, index=False)


def _parse_quality_cols(sub: pd.DataFrame) -> dict:
    reason = sub["unusable_reason"]
    path = sub.get("parse_path")
    return {
        "n_ambiguous": int((reason == "ambiguous").sum()),
        "n_refusal": int((reason == "refusal").sum()),
        "n_no_number": int((reason == "no_number").sum()),
        "n_out_of_range": int((reason == "out_of_range").sum()),
        "n_parse_standalone": int((path == "standalone").sum()) if path is not None else 0,
        "n_parse_fallback": int((path == "fallback_first").sum()) if path is not None else 0,
        "n_parse_insertion_tail": int((path == "insertion_tail").sum()) if path is not None else 0,
    }


def _summarize_nonsocial(df: pd.DataFrame, payoff_max: dict) -> pd.DataFrame:
    rows = []
    for cond, sub in df.groupby("condit"):
        api_err = int((sub["unusable_reason"] == "api_error").sum())
        eligible = sub[sub["unusable_reason"] != "api_error"]
        n = len(sub)
        solved = eligible["decision"].notna()
        over = eligible["unusable_reason"] == "out_of_range"


        comprehensible = int((solved | over).sum())
        n_over = int(over.sum())
        usable = int(solved.sum())
        if comprehensible:
            target = payoff_max.get(cond)
            if target == "x":
                hits = int((eligible.loc[solved, "decision"] == eligible.loc[solved, "stakes"]).sum())
            else:
                hits = int((eligible.loc[solved, "decision"] == float(target or 0)).sum())
            max_rate = hits / comprehensible
        else:
            max_rate = float("nan")
        rows.append({
            "condit": cond,
            "n": n,
            "api_errors": api_err,
            "usable": usable,
            "comprehensible": comprehensible,
            "over_stakes": n_over,
            "over_stakes_rate": (n_over / comprehensible) if comprehensible else float("nan"),
            "unusable_rate": (
                (len(eligible) - usable) / len(eligible) if len(eligible) else float("nan")
            ),
            "payoff_max_rate": max_rate,
            "median_decision": float(eligible.loc[solved, "decision"].median()) if usable else float("nan"),
            "mean_decision":   float(eligible.loc[solved, "decision"].mean())   if usable else float("nan"),
            **_parse_quality_cols(sub),
        })
    return pd.DataFrame(rows).sort_values("condit")


def _summarize_dictator(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, sub in df.groupby("condit"):
        api_err = int((sub["unusable_reason"] == "api_error").sum())
        eligible = sub[sub["unusable_reason"] != "api_error"]
        n = len(sub)
        usable = int(eligible["propshare"].notna().sum())
        propshares = eligible["propshare"].dropna()
        rows.append({
            "condit": cond,
            "n": n,
            "api_errors": api_err,
            "usable": usable,
            "unusable_rate": (
                (len(eligible) - usable) / len(eligible) if len(eligible) else float("nan")
            ),
            "median_propshare": float(propshares.median()) if usable else float("nan"),
            "mean_propshare":   float(propshares.mean())   if usable else float("nan"),
            "p_zero":      float((propshares == 0).mean()) if usable else float("nan"),
            "p_lt01":      float((propshares < 0.01).mean()) if usable else float("nan"),
            "p_half":      float(((propshares - 0.5).abs() < 0.01).mean()) if usable else float("nan"),
            "p_full":      float((propshares == 1).mean()) if usable else float("nan"),
            **_parse_quality_cols(sub),
        })
    return pd.DataFrame(rows).sort_values("condit")


def _plot_nonsocial(df: pd.DataFrame, model: str, exp: str, path: Path) -> None:
    conds = sorted(df["condit"].unique())
    fig, axes = plt.subplots(1, len(conds), figsize=(4.2 * len(conds), 3.2), squeeze=False)
    for ax, cond in zip(axes[0], conds):
        sub = df[df["condit"] == cond]
        usable = sub.dropna(subset=["decision"])
        if not usable.empty:
            ax.scatter(usable["stakes"], usable["decision"], s=4, alpha=0.4)
        ax.plot([0, sub["stakes"].max()], [0, sub["stakes"].max()], "k--", lw=0.8, alpha=0.4)
        ax.set_title(f"{cond} (n={len(sub)}, usable={len(usable)})")
        ax.set_xlabel("stakes (X)")
        ax.set_ylabel("decision")
    fig.suptitle(f"Non-social task — {model} · {exp}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_dictator(
    df: pd.DataFrame,
    model: str,
    exp: str,
    path: Path,
    *,
    xlabel: str = "proportion shared",
) -> None:
    conds = sorted(df["condit"].unique())
    n = len(conds)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows), squeeze=False)
    bins = np.arange(0, 1.05, 0.05)
    for i, cond in enumerate(conds):
        ax = axes[i // cols][i % cols]
        sub = df[df["condit"] == cond]
        propshares = sub["propshare"].dropna()
        ax.hist(propshares.clip(0, 1), bins=bins, color="#4a8")
        median = propshares.median() if not propshares.empty else float("nan")
        ax.axvline(median, color="k", lw=1, ls="--", label=f"median={median:.2f}")
        ax.set_title(f"{cond}\nn={len(sub)} usable={len(propshares)}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("frequency")
        ax.legend(loc="best", fontsize=8)
    for j in range(n, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle(f"{exp} — {model}")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
