from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence


OUTPUT_FIELDS = [
    "source",
    "comparison_scope",
    "metric",
    "direction",
    "task_group",
    "identity_axis",
    "identity",
    "greeting_pattern",
    "coverage_scope",
    "identity_a",
    "identity_b",
    "legacy_mean",
    "modern_mean",
    "delta_modern_minus_legacy",
    "pct_delta",
    "legacy_n_rows",
    "modern_n_rows",
    "legacy_models",
    "modern_models",
]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    value: Callable[[Dict[str, str]], Optional[float]]
    up_label: str
    down_label: str


def _read_csv(path: Optional[str]) -> List[Dict[str, str]]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _field(name: str) -> Callable[[Dict[str, str]], Optional[float]]:
    return lambda row: _float(row.get(name))


def _rate(num_field: str, den_field: str) -> Callable[[Dict[str, str]], Optional[float]]:
    def calc(row: Dict[str, str]) -> Optional[float]:
        num = _float(row.get(num_field))
        den = _float(row.get(den_field))
        if num is None or den is None or den == 0:
            return None
        return num / den

    return calc


def _gap(across_field: str, within_field: str) -> Callable[[Dict[str, str]], Optional[float]]:
    def calc(row: Dict[str, str]) -> Optional[float]:
        across = _float(row.get(across_field))
        within = _float(row.get(within_field))
        if across is None or within is None:
            return None
        return across - within

    return calc


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def _direction(spec: MetricSpec, delta: float) -> str:
    if not math.isfinite(delta) or abs(delta) < 1e-12:
        return "no_clear_change"
    return spec.up_label if delta > 0 else spec.down_label


def _is_legacy(row: Dict[str, str], legacy_provider: str, legacy_model_regex: Optional[re.Pattern]) -> bool:
    provider = str(row.get("provider") or "")
    model = str(row.get("model") or "")
    if provider == legacy_provider:
        return True
    return bool(legacy_model_regex and legacy_model_regex.search(model))


def _canonicalize_premise_pairs(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for row in rows:
        copied = dict(row)
        a = str(copied.get("identity_a") or "")
        b = str(copied.get("identity_b") or "")
        if b and a and b < a:
            copied["identity_a"], copied["identity_b"] = b, a
        out.append(copied)
    return out


def _add_comparisons(
    out_rows: List[Dict[str, object]],
    rows: Sequence[Dict[str, str]],
    source: str,
    scope: str,
    group_fields: Sequence[str],
    metrics: Sequence[MetricSpec],
    legacy_provider: str,
    legacy_model_regex: Optional[re.Pattern],
    min_legacy_rows: int,
    min_modern_rows: int,
) -> None:
    buckets: Dict[tuple, Dict[str, List[tuple[float, Dict[str, str]]]]] = defaultdict(lambda: {"legacy": [], "modern": []})
    for row in rows:
        side = "legacy" if _is_legacy(row, legacy_provider, legacy_model_regex) else "modern"
        group_values = tuple(str(row.get(field) or "") for field in group_fields)
        for spec in metrics:
            value = spec.value(row)
            if value is None:
                continue
            buckets[(spec.name, group_values)][side].append((value, row))

    for (metric_name, group_values), sides in sorted(buckets.items(), key=lambda item: (item[0][0], item[0][1])):
        legacy = sides["legacy"]
        modern = sides["modern"]
        if len(legacy) < min_legacy_rows or len(modern) < min_modern_rows:
            continue
        legacy_values = [value for value, _row in legacy]
        modern_values = [value for value, _row in modern]
        legacy_mean = _mean(legacy_values)
        modern_mean = _mean(modern_values)
        delta = modern_mean - legacy_mean
        pct_delta = delta / abs(legacy_mean) if legacy_mean else math.nan
        spec = next(item for item in metrics if item.name == metric_name)

        group_map = {field: group_values[index] for index, field in enumerate(group_fields)}
        row_out: Dict[str, object] = {field: "" for field in OUTPUT_FIELDS}
        row_out.update(
            {
                "source": source,
                "comparison_scope": scope,
                "metric": metric_name,
                "direction": _direction(spec, delta),
                "legacy_mean": legacy_mean,
                "modern_mean": modern_mean,
                "delta_modern_minus_legacy": delta,
                "pct_delta": pct_delta,
                "legacy_n_rows": len(legacy),
                "modern_n_rows": len(modern),
                "legacy_models": ";".join(sorted({str(row.get("model") or "") for _value, row in legacy})),
                "modern_models": ";".join(sorted({str(row.get("model") or "") for _value, row in modern})),
            }
        )
        for field in ("task_group", "identity_axis", "identity", "greeting_pattern", "coverage_scope", "identity_a", "identity_b"):
            if field in group_map:
                row_out[field] = group_map[field]
        out_rows.append(row_out)


def _styled_greeting_rows(rows: Sequence[Dict[str, str]]) -> List[Dict[str, str]]:
    grouped: Dict[tuple, Dict[str, object]] = {}
    fields = ("provider", "model", "task_group", "identity_axis", "identity_index", "identity")
    for row in rows:
        key = tuple(str(row.get(field) or "") for field in fields)
        bucket = grouped.setdefault(key, {"styled": 0.0, "den": _float(row.get("n_responses")) or 0.0, "row": row})
        if str(row.get("is_identity_styled") or "") in {"1", "true", "True"}:
            bucket["styled"] = float(bucket["styled"]) + (_float(row.get("n_pattern")) or 0.0)
        bucket["den"] = max(float(bucket["den"]), _float(row.get("n_responses")) or 0.0)

    out: List[Dict[str, str]] = []
    for key, bucket in grouped.items():
        base = dict(bucket["row"])
        den = float(bucket["den"])
        styled = float(bucket["styled"])
        base.update(
            {
                "provider": key[0],
                "model": key[1],
                "task_group": key[2],
                "identity_axis": key[3],
                "identity_index": key[4],
                "identity": key[5],
                "greeting_pattern": "identity_styled_total",
                "n_pattern": str(styled),
                "rate": str(styled / den) if den else "",
                "is_identity_styled": "1",
            }
        )
        out.append(base)
    return out


def _write_csv(path: str, rows: Sequence[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare legacy-pkl models with modern models from existing analysis CSVs.")
    parser.add_argument("--flattening-csv", default=None, help="analyze_modern_llm_jsonl.py 的 task×identity 输出。")
    parser.add_argument("--identity-cluster-csv", default=None, help="analyze_modern_llm_jsonl.py --identity-cluster-csv 输出。")
    parser.add_argument("--greeting-csv", default=None, help="analyze_modern_llm_jsonl.py --greeting-csv 输出。")
    parser.add_argument("--coverage-csv", default=None, help="analyze_modern_coverage_premise.py 的 coverage 输出。")
    parser.add_argument("--premise1-csv", default=None, help="analyze_modern_coverage_premise.py 的 premise1 输出。")
    parser.add_argument("--output-csv", default="outputs/modern_legacy_metric_comparison.csv")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--legacy-provider", default="legacy-pkl")
    parser.add_argument("--legacy-model-regex", default=None, help="可选：额外把匹配该正则的 model 当作 legacy。")
    parser.add_argument("--min-legacy-rows", type=int, default=1)
    parser.add_argument("--min-modern-rows", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    legacy_model_regex = re.compile(args.legacy_model_regex) if args.legacy_model_regex else None
    out_rows: List[Dict[str, object]] = []

    diversity_metrics = [
        MetricSpec("unique_ngram_mean", _field("unique_ngram_mean"), "modern_less_flattened_or_more_diverse", "modern_more_flattened_or_less_diverse"),
        MetricSpec("pairwise_cosine_distance_mean", _field("pairwise_cosine_distance_mean"), "modern_less_flattened_or_more_diverse", "modern_more_flattened_or_less_diverse"),
        MetricSpec("embedding_covariance_trace", _field("embedding_covariance_trace"), "modern_less_flattened_or_more_diverse", "modern_more_flattened_or_less_diverse"),
        MetricSpec("mc_unique", _field("mc_unique"), "modern_less_flattened_or_more_diverse", "modern_more_flattened_or_less_diverse"),
        MetricSpec("ai_disclaimer_rate", _rate("n_ai_disclaimer", "n_responses"), "modern_more_ai_disclaimer", "modern_less_ai_disclaimer"),
    ]
    coverage_metrics = [
        MetricSpec("vendi_point", _field("vendi_point"), "modern_more_coverage", "modern_less_coverage"),
        MetricSpec("det_point", _field("det_point"), "modern_more_coverage", "modern_less_coverage"),
        MetricSpec("mc_unique", _field("mc_unique"), "modern_more_coverage", "modern_less_coverage"),
    ]
    greeting_metrics = [
        MetricSpec("greeting_rate", _field("rate"), "modern_more_greeting_pattern", "modern_less_greeting_pattern"),
    ]
    premise_metrics = [
        MetricSpec("distance_gap_across_minus_within", _gap("mean_across", "mean_within"), "modern_stronger_identity_effect", "modern_weaker_identity_effect"),
        MetricSpec(
            "persample_gap_across_minus_within",
            _gap("mean_across_persample", "mean_within_persample"),
            "modern_stronger_identity_effect",
            "modern_weaker_identity_effect",
        ),
    ]

    flattening_rows = _read_csv(args.flattening_csv)
    if flattening_rows:
        _add_comparisons(
            out_rows,
            flattening_rows,
            "flattening",
            "task_identity",
            ["task_group", "identity_axis", "identity"],
            diversity_metrics,
            args.legacy_provider,
            legacy_model_regex,
            args.min_legacy_rows,
            args.min_modern_rows,
        )

    identity_cluster_rows = _read_csv(args.identity_cluster_csv)
    if identity_cluster_rows:
        _add_comparisons(
            out_rows,
            identity_cluster_rows,
            "flattening",
            "identity_cluster_fig4",
            ["identity_axis", "identity"],
            diversity_metrics,
            args.legacy_provider,
            legacy_model_regex,
            args.min_legacy_rows,
            args.min_modern_rows,
        )

    coverage_rows = _read_csv(args.coverage_csv)
    if coverage_rows:
        _add_comparisons(
            out_rows,
            coverage_rows,
            "coverage",
            "task_axis_scope",
            ["task_group", "identity_axis", "coverage_scope"],
            coverage_metrics,
            args.legacy_provider,
            legacy_model_regex,
            args.min_legacy_rows,
            args.min_modern_rows,
        )

    greeting_rows = _read_csv(args.greeting_csv)
    if greeting_rows:
        _add_comparisons(
            out_rows,
            greeting_rows + _styled_greeting_rows(greeting_rows),
            "essentialization",
            "task_identity_greeting_pattern",
            ["task_group", "identity_axis", "identity", "greeting_pattern"],
            greeting_metrics,
            args.legacy_provider,
            legacy_model_regex,
            args.min_legacy_rows,
            args.min_modern_rows,
        )

    premise_rows = _canonicalize_premise_pairs(_read_csv(args.premise1_csv))
    if premise_rows:
        _add_comparisons(
            out_rows,
            premise_rows,
            "premise1",
            "task_axis_identity_pair",
            ["task_group", "identity_axis", "identity_a", "identity_b"],
            premise_metrics,
            args.legacy_provider,
            legacy_model_regex,
            args.min_legacy_rows,
            args.min_modern_rows,
        )

    if not any((flattening_rows, identity_cluster_rows, coverage_rows, greeting_rows, premise_rows)):
        raise SystemExit("No input CSVs provided.")

    out_rows.sort(key=lambda row: tuple(str(row.get(field)) for field in ("source", "comparison_scope", "metric", "task_group", "identity_axis", "identity", "greeting_pattern", "coverage_scope", "identity_a", "identity_b")))
    _write_csv(args.output_csv, out_rows)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"n_rows": len(out_rows), "rows": out_rows}, handle, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_rows)} comparison rows -> {args.output_csv}")
    if args.output_json:
        print(f"Wrote JSON comparison -> {args.output_json}")


if __name__ == "__main__":
    main()
