import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "scripts", "shared")
for _p in (PROJECT_ROOT, SHARED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis_common as common


GROUP_FIELDS = ["provider", "model", "task_key", "identity_axis", "identity_index", "identity"]
SUMMARY_FIELDS = [
    "provider",
    "model",
    "task_key",
    "task_group",
    "identity_axis",
    "identity_index",
    "identity",
    "n_responses",
    "n_empty",
    "n_ai_disclaimer",
    "mean_chars",
    "mean_words",
    "unique_ngram_mean",
    "unique_ngram_ci025",
    "unique_ngram_ci085",
    "unique_ngram_ci915",
    "unique_ngram_ci975",
    "pairwise_cosine_distance_mean",
    "pairwise_cosine_distance_ci025",
    "pairwise_cosine_distance_ci085",
    "pairwise_cosine_distance_ci915",
    "pairwise_cosine_distance_ci975",
    "embedding_covariance_trace",
    "embedding_covariance_trace_ci025",
    "embedding_covariance_trace_ci085",
    "embedding_covariance_trace_ci915",
    "embedding_covariance_trace_ci975",
    "embedding_backend",
    "mc_unique",
]

IDENTITY_FIELDS = ["provider", "model", "identity_axis", "identity_index", "identity"]
IDENTITY_CLUSTER_FIELDS = [
    "provider",
    "model",
    "task_scope",
    "task_clusters",
    "n_task_clusters",
    "identity_axis",
    "identity_index",
    "identity",
    "n_responses",
    "n_empty",
    "n_ai_disclaimer",
    "mean_chars",
    "mean_words",
    "unique_ngram_mean",
    "unique_ngram_ci025",
    "unique_ngram_ci085",
    "unique_ngram_ci915",
    "unique_ngram_ci975",
    "pairwise_cosine_distance_mean",
    "pairwise_cosine_distance_ci025",
    "pairwise_cosine_distance_ci085",
    "pairwise_cosine_distance_ci915",
    "pairwise_cosine_distance_ci975",
    "embedding_covariance_trace",
    "embedding_covariance_trace_ci025",
    "embedding_covariance_trace_ci085",
    "embedding_covariance_trace_ci915",
    "embedding_covariance_trace_ci975",
    "embedding_backend",
    "mc_unique",
]

GREETING_FIELDS = [
    "provider",
    "model",
    "task_group",
    "identity_axis",
    "identity_index",
    "identity",
    "n_responses",
    "greeting_pattern",
    "n_pattern",
    "rate",
    "is_identity_styled",
    "example_opening",
]


TOKEN_RE = re.compile(r"(?u)\b\w+\b")


_AI_PHRASES = (" ai ", "as an ai", "language model", "i'm sorry, but i can't assist", "openai")

_GREETING_RULES = [
    ("hey_girl", re.compile(r"^(?:hey|hi|hello)\s+(?:girl|girls|girlfriend|lady|ladies|sis|sister)\b", re.I), True),
    ("hey_buddy", re.compile(r"^(?:hey|hi|hello)\s+(?:buddy|pal|friend|folks?)\b", re.I), True),
    ("hey_dude", re.compile(r"^(?:hey|hi|hello)\s+(?:dude|bro|brother)\b", re.I), True),
    ("hey_man", re.compile(r"^(?:hey|hi|hello)\s+(?:man|men|sir|mister)\b", re.I), True),
    ("dear_group", re.compile(r"^dear\s+[a-z][a-z'\-]+", re.I), True),
    ("generic_hey", re.compile(r"^hey\b", re.I), False),
    ("generic_hi", re.compile(r"^hi\b", re.I), False),
    ("generic_hello", re.compile(r"^hello\b", re.I), False),
    ("generic_dear", re.compile(r"^dear\b", re.I), False),
]


def _has_ai_disclaimer(text: str) -> bool:
    low = f" {text.lower()} "
    return any(p in low for p in _AI_PHRASES)


class AnalysisError(RuntimeError):
    pass


def load_jsonl(paths: Sequence[str]) -> List[Dict]:
    records = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise AnalysisError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
                if "response" not in item:
                    continue
                item.setdefault("provider", "jsonl")
                item.setdefault("model", "unknown")
                item.setdefault("task_key", item.get("task", "unknown"))
                item.setdefault("identity_axis", "unknown")
                item.setdefault("identity_index", None)
                item.setdefault("identity", "unknown")
                records.append(item)
    return records


def _split_r2b_text(text: str, version: int) -> Optional[List[str]]:
    if version == 0:
        sets = [["P1", "P2", "P3"], ["1.", "2.", "3."], ["1)", "2)", "3)"],
                ["first phrase", "second phrase", "third phrase"], ["1", "2", "3"]]
    else:
        sets = [["P3", "P4", "P5"], ["3.", "4.", "5."], ["3)", "4)", "5)"],
                ["third phrase", "fourth phrase", "fifth phrase"], ["3", "4", "5"]]
    chosen = next((s for s in sets if all(d in text for d in s)), None)
    if chosen is None:


        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if len(paras) == 3:
            return paras
        return None
    parts: List[str] = []
    for i, delim in enumerate(chosen):
        start = text.index(delim) + len(delim)
        end = text.index(chosen[i + 1]) if i + 1 < len(chosen) else len(text)
        chunk = text[start:end].strip()
        if chunk[:2] in (": ", ". "):
            chunk = chunk[2:].strip()
        if len(chunk) >= 2 and chunk[0] in "\"'" and chunk[-1] == chunk[0]:
            chunk = chunk[1:-1].strip()
        parts.append(chunk)
    return parts


def expand_r2b_records(records: Sequence[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for item in records:
        task_key = str(item.get("task_key") or "")
        version = common.r2b_split_version(task_key)
        if version is None:
            out.append(item)
            continue
        parts = _split_r2b_text(clean_response(item.get("response", "")), version)
        if not parts:
            out.append(item)
            continue
        mc_list = item.get("mc")
        for sub, chunk in enumerate(parts):
            new = dict(item)
            new["response"] = chunk
            new["task_key"] = f"{task_key}_{sub}"
            new["r2b_sub_index"] = sub
            if isinstance(mc_list, list) and sub < len(mc_list):
                new["mc"] = mc_list[sub]
            out.append(new)
    return out


def group_records(records: Sequence[Dict], min_responses: int, per_group_cap: Optional[int]) -> Dict[Tuple, List[Dict]]:
    groups = defaultdict(list)
    for item in records:
        key = tuple(item.get(field) for field in GROUP_FIELDS)
        groups[key].append(item)
    out: Dict[Tuple, List[Dict]] = {}
    for key, value in groups.items():
        if len(value) < min_responses:
            continue
        value = sorted(value, key=lambda x: x.get("sample_index", 0))


        if per_group_cap:
            value = value[:per_group_cap]
        out[key] = value
    return out


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def ngrams(tokens: Sequence[str], n: int) -> List[str]:
    if len(tokens) < n:
        return []
    return [" ".join(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]


def unique_ngram_props(texts: Sequence[str]) -> List[float]:
    per_text = []
    counts = defaultdict(int)
    threshold = int(0.05 * len(texts))
    for text in texts:
        features = set(ngrams(tokenize(text), 1) + ngrams(tokenize(text), 2))
        per_text.append(features)
        for feature in features:
            counts[feature] += 1
    props = []
    for features in per_text:
        if not features:
            continue
        unique_count = sum(1 for feature in features if counts[feature] <= 1 + threshold)
        props.append(unique_count / len(features))
    return props


def ci_summary(values: Sequence[float], bootstrap: int, rng: np.random.Generator) -> Tuple[float, float, float, float, float]:
    values = np.array([value for value in values if not math.isnan(float(value))], dtype=float)
    if len(values) == 0:
        return (math.nan, math.nan, math.nan, math.nan, math.nan)
    point = float(np.mean(values))
    if bootstrap <= 0 or len(values) == 1:
        return (point, math.nan, math.nan, math.nan, math.nan)
    boot = []
    for _ in range(bootstrap):
        indices = rng.integers(0, len(values), size=len(values))
        boot.append(float(np.mean(values[indices])))
    boot = np.sort(np.array(boot))
    return (
        point,
        percentile_from_sorted(boot, 2.5),
        percentile_from_sorted(boot, 8.5),
        percentile_from_sorted(boot, 91.5),
        percentile_from_sorted(boot, 97.5),
    )


def percentile_from_sorted(values: np.ndarray, pct: float) -> float:
    if len(values) == 0:
        return math.nan
    index = int(round((pct / 100) * (len(values) - 1)))
    index = max(0, min(index, len(values) - 1))
    return float(values[index])


def fit_embeddings(groups: Dict[Tuple, List[Dict]], backend: str, model_name: str, batch_size: int) -> Tuple[Dict[Tuple, np.ndarray], str]:
    texts_by_key = {key: [clean_response(item.get("response", "")) for item in items] for key, items in groups.items()}
    all_texts = [text for texts in texts_by_key.values() for text in texts]
    if backend == "none":
        return {}, "none"
    if backend == "auto":
        backend = "sbert"
    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            backend = "tfidf"
        else:
            model = SentenceTransformer(model_name)
            embeddings_by_key = {}
            for key, texts in texts_by_key.items():
                embeddings_by_key[key] = np.array(model.encode(texts, batch_size=batch_size, show_progress_bar=False))
            return embeddings_by_key, "sbert"
    if backend == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise AnalysisError("Install scikit-learn or sentence-transformers to compute embedding metrics") from exc
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, lowercase=True)
        matrix = vectorizer.fit_transform(all_texts)
        embeddings_by_key = {}
        cursor = 0
        for key, texts in texts_by_key.items():
            embeddings_by_key[key] = matrix[cursor : cursor + len(texts)]
            cursor += len(texts)
        return embeddings_by_key, "tfidf"
    raise AnalysisError(f"Unknown embedding backend: {backend}")


def clean_response(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def cosine_distance_values(embeddings: np.ndarray) -> np.ndarray:
    n_rows = embeddings.shape[0] if hasattr(embeddings, "shape") else len(embeddings)
    if n_rows < 2:
        return np.array([], dtype=float)
    if hasattr(embeddings, "tocsr"):
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(embeddings)
        upper = np.triu_indices(embeddings.shape[0], k=1)
        return distances[upper]
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = np.divide(embeddings, norms, out=np.zeros_like(embeddings, dtype=float), where=norms != 0)
    sims = safe @ safe.T
    distances = 1.0 - sims
    upper = np.triu_indices(len(embeddings), k=1)
    return distances[upper]


def covariance_trace(embeddings) -> float:
    n_rows = embeddings.shape[0] if hasattr(embeddings, "shape") else len(embeddings)
    if n_rows < 2:
        return math.nan
    if hasattr(embeddings, "tocsr"):
        mean = np.asarray(embeddings.mean(axis=0)).ravel()
        mean_square = np.asarray(embeddings.multiply(embeddings).mean(axis=0)).ravel()

        variance = mean_square - mean**2
        return float(np.maximum(variance, 0).sum())
    return float(np.var(embeddings, axis=0, ddof=0).sum())


def bootstrap_covariance_trace(embeddings: np.ndarray, bootstrap: int, rng: np.random.Generator) -> Tuple[float, float, float, float, float]:
    point = covariance_trace(embeddings)
    n_rows = embeddings.shape[0] if hasattr(embeddings, "shape") else len(embeddings)
    if bootstrap <= 0 or n_rows < 2:
        return (point, math.nan, math.nan, math.nan, math.nan)
    values = []
    for _ in range(bootstrap):
        indices = rng.integers(0, n_rows, size=n_rows)
        values.append(covariance_trace(embeddings[indices]))
    values = np.sort(np.array(values, dtype=float))

    return (
        point,
        percentile_from_sorted(values, 2.5),
        percentile_from_sorted(values, 8.5),
        percentile_from_sorted(values, 91.5),
        percentile_from_sorted(values, 97.5),
    )


def _unique_mcs(items: Sequence[Dict]) -> float:
    vals = set()
    for item in items:
        v = item.get("mc")
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)) and 1 <= int(v) <= 5:
            vals.add(int(v))
    return float(len(vals)) if vals else math.nan


def summarize_group(key: Tuple, items: Sequence[Dict], embeddings: Optional[np.ndarray], embedding_backend: str, bootstrap: int, rng: np.random.Generator) -> Dict:
    texts = [clean_response(item.get("response", "")) for item in items]
    words = [len(tokenize(text)) for text in texts]
    chars = [len(text) for text in texts]
    n_empty = sum(1 for text in texts if not text)
    unique_point, unique_ci025, unique_ci085, unique_ci915, unique_ci975 = ci_summary(unique_ngram_props(texts), bootstrap, rng)
    if embeddings is None:
        pairwise = (math.nan, math.nan, math.nan, math.nan, math.nan)
        trace = (math.nan, math.nan, math.nan, math.nan, math.nan)
    else:
        pairwise = ci_summary(cosine_distance_values(embeddings), bootstrap, rng)
        trace = bootstrap_covariance_trace(embeddings, bootstrap, rng)
    row = dict(zip(GROUP_FIELDS, key))
    row.update(
        {
            "task_group": common.task_group(str(row.get("task_key") or "")),
            "n_responses": len(texts),
            "n_empty": n_empty,
            "n_ai_disclaimer": sum(1 for text in texts if _has_ai_disclaimer(text)),
            "mean_chars": float(np.mean(chars)) if chars else math.nan,
            "mean_words": float(np.mean(words)) if words else math.nan,
            "unique_ngram_mean": unique_point,
            "unique_ngram_ci025": unique_ci025,
            "unique_ngram_ci085": unique_ci085,
            "unique_ngram_ci915": unique_ci915,
            "unique_ngram_ci975": unique_ci975,
            "pairwise_cosine_distance_mean": pairwise[0],
            "pairwise_cosine_distance_ci025": pairwise[1],
            "pairwise_cosine_distance_ci085": pairwise[2],
            "pairwise_cosine_distance_ci915": pairwise[3],
            "pairwise_cosine_distance_ci975": pairwise[4],
            "embedding_covariance_trace": trace[0],
            "embedding_covariance_trace_ci025": trace[1],
            "embedding_covariance_trace_ci085": trace[2],
            "embedding_covariance_trace_ci915": trace[3],
            "embedding_covariance_trace_ci975": trace[4],
            "embedding_backend": embedding_backend,
            "mc_unique": _unique_mcs(items),
        }
    )
    return row


def _finite_float(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _mean_unique_ngram(items: Sequence[Dict]) -> float:
    texts = [clean_response(item.get("response", "")) for item in items]
    props = unique_ngram_props(texts)
    return float(np.mean(props)) if props else math.nan


def _embedding_matrix_for_items(items: Sequence[Dict], embedding_lookup: Dict[int, object]):
    rows = [embedding_lookup[id(item)] for item in items if id(item) in embedding_lookup]
    if not rows:
        return None
    first = rows[0]
    if hasattr(first, "tocsr"):
        from scipy.sparse import vstack

        return vstack(rows)
    return np.vstack(rows).astype(float)


def _mean_pairwise_distance(items: Sequence[Dict], embedding_lookup: Dict[int, object]) -> float:
    embeddings = _embedding_matrix_for_items(items, embedding_lookup)
    if embeddings is None:
        return math.nan
    values = cosine_distance_values(embeddings)
    return float(np.mean(values)) if values.size else math.nan


def _covariance_trace_for_items(items: Sequence[Dict], embedding_lookup: Dict[int, object]) -> float:
    embeddings = _embedding_matrix_for_items(items, embedding_lookup)
    if embeddings is None:
        return math.nan
    return covariance_trace(embeddings)


def _cluster_bootstrap_summary(
    clusters: Dict[str, List[Dict]],
    compute,
    bootstrap: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    cluster_keys = sorted(clusters)
    all_items: List[Dict] = []
    for key in cluster_keys:
        all_items.extend(clusters[key])
    point = compute(all_items)
    if bootstrap <= 0 or len(cluster_keys) < 2:
        return (point, math.nan, math.nan, math.nan, math.nan)
    values: List[float] = []
    for _ in range(bootstrap):
        sampled: List[Dict] = []
        for index in rng.integers(0, len(cluster_keys), size=len(cluster_keys)):
            sampled.extend(clusters[cluster_keys[int(index)]])
        value = _finite_float(compute(sampled))
        if value is not None:
            values.append(value)
    if not values:
        return (point, math.nan, math.nan, math.nan, math.nan)
    values_arr = np.sort(np.asarray(values, dtype=float))
    return (
        point,
        percentile_from_sorted(values_arr, 2.5),
        percentile_from_sorted(values_arr, 8.5),
        percentile_from_sorted(values_arr, 91.5),
        percentile_from_sorted(values_arr, 97.5),
    )


def build_embedding_lookup(groups: Dict[Tuple, List[Dict]], embeddings_by_key: Dict[Tuple, np.ndarray], backend: str) -> Dict[int, object]:
    if backend == "none":
        return {}
    lookup: Dict[int, object] = {}
    for key, items in groups.items():
        embeddings = embeddings_by_key.get(key)
        if embeddings is None:
            continue
        for index, item in enumerate(items):
            lookup[id(item)] = embeddings[index]
    return lookup


def group_records_by_identity(groups: Dict[Tuple, List[Dict]]) -> Dict[Tuple, List[Dict]]:
    out: Dict[Tuple, List[Dict]] = defaultdict(list)
    for items in groups.values():
        for item in items:
            key = tuple(item.get(field) for field in IDENTITY_FIELDS)
            out[key].append(item)
    return out


def summarize_identity_cluster(
    key: Tuple,
    items: Sequence[Dict],
    embedding_lookup: Dict[int, object],
    embedding_backend: str,
    bootstrap: int,
    min_task_clusters: int,
    rng: np.random.Generator,
) -> Optional[Dict]:
    clusters: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        clusters[common.task_group(str(item.get("task_key") or ""))].append(item)
    if len(clusters) < min_task_clusters:
        return None

    texts = [clean_response(item.get("response", "")) for item in items]
    words = [len(tokenize(text)) for text in texts]
    chars = [len(text) for text in texts]
    unique = _cluster_bootstrap_summary(clusters, _mean_unique_ngram, bootstrap, rng)
    if embedding_backend == "none" or not embedding_lookup:
        pairwise = (math.nan, math.nan, math.nan, math.nan, math.nan)
        trace = (math.nan, math.nan, math.nan, math.nan, math.nan)
    else:
        pairwise = _cluster_bootstrap_summary(
            clusters,
            lambda sampled: _mean_pairwise_distance(sampled, embedding_lookup),
            bootstrap,
            rng,
        )
        trace = _cluster_bootstrap_summary(
            clusters,
            lambda sampled: _covariance_trace_for_items(sampled, embedding_lookup),
            bootstrap,
            rng,
        )

    row = dict(zip(IDENTITY_FIELDS, key))
    row.update(
        {
            "task_scope": "all_task_groups_clustered",
            "task_clusters": ";".join(sorted(clusters)),
            "n_task_clusters": len(clusters),
            "n_responses": len(texts),
            "n_empty": sum(1 for text in texts if not text),
            "n_ai_disclaimer": sum(1 for text in texts if _has_ai_disclaimer(text)),
            "mean_chars": float(np.mean(chars)) if chars else math.nan,
            "mean_words": float(np.mean(words)) if words else math.nan,
            "unique_ngram_mean": unique[0],
            "unique_ngram_ci025": unique[1],
            "unique_ngram_ci085": unique[2],
            "unique_ngram_ci915": unique[3],
            "unique_ngram_ci975": unique[4],
            "pairwise_cosine_distance_mean": pairwise[0],
            "pairwise_cosine_distance_ci025": pairwise[1],
            "pairwise_cosine_distance_ci085": pairwise[2],
            "pairwise_cosine_distance_ci915": pairwise[3],
            "pairwise_cosine_distance_ci975": pairwise[4],
            "embedding_covariance_trace": trace[0],
            "embedding_covariance_trace_ci025": trace[1],
            "embedding_covariance_trace_ci085": trace[2],
            "embedding_covariance_trace_ci915": trace[3],
            "embedding_covariance_trace_ci975": trace[4],
            "embedding_backend": embedding_backend,
            "mc_unique": _unique_mcs(items),
        }
    )
    return row


def _opening_fragment(text: str) -> str:
    text = clean_response(text)
    text = text.lstrip(" \t\r\n\"'`“”‘’")
    text = re.sub(r"\s+", " ", text)
    if not text:
        return ""
    match = re.match(r"^(.{0,100}?)(?:[.!?\n]|$)", text)
    return (match.group(1) if match else text[:100]).strip(" ,;:-")


def _classify_greeting(text: str) -> Tuple[str, str, bool]:
    opening = _opening_fragment(text)
    normalized = opening.lower()
    for name, pattern, styled in _GREETING_RULES:
        match = pattern.search(normalized)
        if match:
            return name, opening[:100], styled
    return ("none", opening[:100], False)


def summarize_greetings(items: Sequence[Dict]) -> List[Dict]:
    grouped: Dict[Tuple, List[Dict]] = defaultdict(list)
    for item in items:
        key = (
            item.get("provider"),
            item.get("model"),
            common.task_group(str(item.get("task_key") or "")),
            item.get("identity_axis"),
            item.get("identity_index"),
            item.get("identity"),
        )
        grouped[key].append(item)

    rows: List[Dict] = []
    for key, group_items in grouped.items():
        counts: Counter = Counter()
        examples: Dict[str, str] = {}
        styled_flags: Dict[str, bool] = {}
        for item in group_items:
            pattern, opening, styled = _classify_greeting(item.get("response", ""))
            counts[pattern] += 1
            examples.setdefault(pattern, opening)
            styled_flags[pattern] = styled
        denom = len(group_items)
        for pattern, count in sorted(counts.items()):
            row = dict(zip(GREETING_FIELDS[:6], key))
            row.update(
                {
                    "n_responses": denom,
                    "greeting_pattern": pattern,
                    "n_pattern": count,
                    "rate": count / denom if denom else math.nan,
                    "is_identity_styled": int(bool(styled_flags.get(pattern))),
                    "example_opening": examples.get(pattern, ""),
                }
            )
            rows.append(row)
    rows.sort(key=lambda row: tuple(str(row.get(field)) for field in GREETING_FIELDS[:7] + ["greeting_pattern"]))
    return rows


def write_csv_with_fields(path: str, fields: Sequence[str], rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def write_csv(path: str, rows: Sequence[Dict]) -> None:
    write_csv_with_fields(path, SUMMARY_FIELDS, rows)


def write_json(path: str, rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(list(rows), handle, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze modern LLM JSONL generations with standalone diversity/flattening metrics.")
    parser.add_argument("--input", nargs="+", default=[])
    parser.add_argument("--legacy-pkl", default=None)
    parser.add_argument("--output-csv", default="modern_llm_diversity_summary.csv")
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--identity-cluster-csv",
        default=None,
        help="可选：输出论文 Fig.4 口径的身份级跨题 cluster bootstrap 汇总。"
    )
    parser.add_argument(
        "--identity-cluster-min-tasks",
        type=int,
        default=3,
        help="身份级 cluster bootstrap 至少需要的题目/子题 cluster 数（论文 Fig.4 通常为 3-6）。",
    )
    parser.add_argument(
        "--greeting-csv",
        default=None,
        help="可选：输出开场问候语模式频率（如 hey_girl / hey_buddy），用于轻量本质化检查。"
    )
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf", "none"], default="auto")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-responses", type=int, default=2)
    parser.add_argument("--per-group-cap", type=int, default=None,
                        help="每组最多取前 N 条（按 sample_index 序）；跨模型比较请统一（论文 n=100）。")
    parser.add_argument("--no-identity-clean", action="store_true",
                        help="关闭身份词清洗。论文 Methods 对所有分析先移除显式身份词，默认开启。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = []
    if args.input:
        records.extend(load_jsonl(args.input))
    if args.legacy_pkl:
        records.extend(common.load_legacy_generations(args.legacy_pkl))
    if not records:
        raise AnalysisError("No records loaded. Provide --input JSONL files or --legacy-pkl.")
    records = expand_r2b_records(records)
    if not args.no_identity_clean:

        common.clean_records(records)
    groups = group_records(records, args.min_responses, args.per_group_cap)
    if not groups:
        raise AnalysisError(f"No groups have at least {args.min_responses} responses.")
    print(f"Loaded {len(records)} records into {len(groups)} analyzable groups")
    if args.dry_run:
        for key, items in list(groups.items())[:10]:
            preview = dict(zip(GROUP_FIELDS, key))
            preview["n_responses"] = len(items)
            print(json.dumps(preview, ensure_ascii=False))
        return
    embeddings_by_key, actual_backend = fit_embeddings(groups, args.embedding_backend, args.sbert_model, args.batch_size)
    rng = np.random.default_rng(args.seed)
    rows = []
    for key, items in groups.items():
        embeddings = embeddings_by_key.get(key) if actual_backend != "none" else None
        rows.append(summarize_group(key, items, embeddings, actual_backend, args.bootstrap, rng))
    rows = sorted(rows, key=lambda row: tuple(str(row.get(field)) for field in GROUP_FIELDS))
    write_csv(args.output_csv, rows)

    analysis_items = [item for items in groups.values() for item in items]
    if args.identity_cluster_csv:
        embedding_lookup = build_embedding_lookup(groups, embeddings_by_key, actual_backend)
        cluster_rng = np.random.default_rng(args.seed + 1000003)
        cluster_rows = []
        for key, items in group_records_by_identity(groups).items():
            row = summarize_identity_cluster(
                key,
                items,
                embedding_lookup,
                actual_backend,
                args.bootstrap,
                args.identity_cluster_min_tasks,
                cluster_rng,
            )
            if row is not None:
                cluster_rows.append(row)
        cluster_rows.sort(key=lambda row: tuple(str(row.get(field)) for field in IDENTITY_FIELDS))
        write_csv_with_fields(args.identity_cluster_csv, IDENTITY_CLUSTER_FIELDS, cluster_rows)

    if args.greeting_csv:
        greeting_rows = summarize_greetings(analysis_items)
        write_csv_with_fields(args.greeting_csv, GREETING_FIELDS, greeting_rows)

    if args.output_json:
        write_json(args.output_json, rows)
    print(f"Wrote {len(rows)} summary rows to {args.output_csv}")
    if args.identity_cluster_csv:
        print(f"Wrote identity-level cluster bootstrap rows to {args.identity_cluster_csv}")
    if args.greeting_csv:
        print(f"Wrote greeting-pattern rows to {args.greeting_csv}")
    if args.output_json:
        print(f"Wrote JSON summary to {args.output_json}")
    print(f"Embedding backend: {actual_backend}")


if __name__ == "__main__":
    main()
