from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from collections import defaultdict
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SHARED_DIR = os.path.join(PROJECT_ROOT, "scripts", "shared")
for _p in (PROJECT_ROOT, SHARED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import analysis_common as common


AXIS_FIELDS = ["provider", "model", "task_key", "identity_axis"]

COVERAGE_FIELDS = [
    "provider",
    "model",
    "task_key",
    "task_group",
    "identity_axis",
    "coverage_scope",
    "n_identities",
    "n_total",
    "embedding_backend",
    "embedding_dim",
    "vendi_point",
    "vendi_ci025",
    "vendi_ci085",
    "vendi_ci915",
    "vendi_ci975",
    "det_point",
    "det_ci025",
    "det_ci085",
    "det_ci915",
    "det_ci975",
    "det_pca",
    "mc_unique",
]

PREMISE1_FIELDS = [
    "provider",
    "model",
    "task_key",
    "task_group",
    "identity_axis",
    "identity_a",
    "identity_b",
    "n_a",
    "n_b",
    "mean_within",
    "mean_across",
    "t_stat",
    "p_value",

    "mean_within_persample",
    "mean_across_persample",
    "wilcoxon_stat",
    "wilcoxon_p",

    "chisq",
    "chisq_p",
]


class AnalysisError(RuntimeError):
    pass


def clean_response(value) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_jsonl(paths: Sequence[str]) -> List[Dict]:
    records: List[Dict] = []
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
                if not clean_response(item.get("response")):
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


def group_by_axis(records: Sequence[Dict]) -> Dict[Tuple, List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for item in records:
        key = tuple(item.get(field) for field in AXIS_FIELDS)
        groups[key].append(item)
    return groups


def group_by_identity(items: Sequence[Dict]) -> Dict[Tuple, List[Dict]]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for item in items:
        key = (item.get("identity_index"), item.get("identity"))
        groups[key].append(item)
    return {
        key: sorted(value, key=lambda x: x.get("sample_index", 0))
        for key, value in groups.items()
    }


def embed_all(records: Sequence[Dict], backend: str, model_name: str, batch_size: int):
    texts = [clean_response(item.get("response", "")) for item in records]
    if backend == "auto":
        backend = "sbert"
    if backend == "sbert":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise AnalysisError(
                "缺少 sentence-transformers，无法用 SBERT 嵌入（det 需要它以复用 fitted_pca.pkl）。"
                "请先 pip install sentence-transformers，或改用 --embedding-backend tfidf 配合 --self-pca。"
            ) from exc
        model = SentenceTransformer(model_name)
        matrix = np.asarray(model.encode(texts, batch_size=batch_size, show_progress_bar=False), dtype=float)
        return matrix, "sbert"
    if backend == "tfidf":
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise AnalysisError("缺少 scikit-learn，无法计算 tfidf 嵌入。") from exc
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, lowercase=True)
        matrix = vectorizer.fit_transform(texts).toarray().astype(float)
        return matrix, "tfidf"
    raise AnalysisError(f"未知 embedding backend：{backend}")


def cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    from sklearn.metrics.pairwise import cosine_distances

    return cosine_distances(embeddings)


def vendi_from_distance_kernel(kernel: np.ndarray) -> float:
    n = kernel.shape[0]
    if n < 2:
        return math.nan
    eigenvalues = np.linalg.eigvalsh(kernel / n)
    positive = eigenvalues[eigenvalues > 1e-12]
    if positive.size == 0:
        return math.nan
    return float(np.exp(-np.sum(positive * np.log(positive))))


def covariance_det(embeddings: np.ndarray, pca) -> float:
    from sklearn.covariance import empirical_covariance

    reduced = pca.transform(embeddings)
    return float(np.linalg.det(empirical_covariance(reduced)))


def percentile_from_sorted(values: np.ndarray, pct: float) -> float:
    if len(values) == 0:
        return math.nan
    index = int(round((pct / 100) * (len(values) - 1)))
    index = max(0, min(index, len(values) - 1))
    return float(values[index])


def _valid_mc(value) -> bool:
    return (not isinstance(value, bool)) and isinstance(value, (int, float)) and 1 <= int(value) <= 5


def _unique_mcs(items: Sequence[Dict]) -> float:
    vals = {int(it.get("mc")) for it in items if _valid_mc(it.get("mc"))}
    return float(len(vals)) if vals else math.nan


def _chisq_homogeneity(mcs_a: Sequence[int], mcs_b: Sequence[int]):
    from scipy.stats import chi2

    a, b = np.asarray(mcs_a), np.asarray(mcs_b)
    ca = np.array([np.sum(a == i) for i in range(1, 6)], dtype=float)
    cb = np.array([np.sum(b == i) for i in range(1, 6)], dtype=float)
    both0 = (ca == 0) & (cb == 0)
    ca[both0] += 1
    cb[both0] += 1
    observed = np.vstack([ca, cb]).T
    row_tot = observed.sum(axis=1, keepdims=True)
    col_tot = observed.sum(axis=0, keepdims=True)
    n = observed.sum()
    expected = row_tot @ col_tot / n
    chisq = float(np.sum((observed - expected) ** 2 / expected))
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return chisq, float(1 - chi2.cdf(chisq, dof))


def load_pca(pca_path: Optional[str], self_pca: bool, embeddings: np.ndarray, n_components: int):
    if self_pca or not pca_path:
        from sklearn.decomposition import PCA

        n = min(n_components, embeddings.shape[0], embeddings.shape[1])
        pca = PCA(n_components=n)
        pca.fit(embeddings)
        return pca, f"self-fit({n})"
    if not os.path.exists(pca_path):
        raise AnalysisError(
            f"找不到 PCA 文件：{pca_path}。请确认 files/fitted_pca.pkl 存在，或加 --self-pca 改为自 fit。"
        )
    with open(pca_path, "rb") as handle:
        pca = pickle.load(handle)
    return pca, os.path.basename(pca_path)


def _capped(items: List[Dict], cap: Optional[int]) -> List[Dict]:
    return items[:cap] if cap else items


def _coverage_scopes(axis: str, identity_groups: Dict[Tuple, List[Dict]]):
    if axis != "other":
        return [(axis, identity_groups)]
    by_cat: Dict[str, Dict[Tuple, List[Dict]]] = defaultdict(dict)
    for key, items in identity_groups.items():
        by_cat[common.other_category(key[1])][key] = items
    return sorted(by_cat.items())


def analyze_coverage(
    axis_key: Tuple,
    scope: str,
    identity_groups: Dict[Tuple, List[Dict]],
    embeddings_index: Dict[int, int],
    embedding_matrix: np.ndarray,
    backend: str,
    pca,
    pca_label: str,
    bootstrap: int,
    rng: np.random.Generator,
    per_identity_cap: Optional[int],
    triplet_iterations: int,
) -> Optional[Dict]:

    def emb_of(members: List[Dict]) -> np.ndarray:
        rows = [embeddings_index[id(item)] for item in members]
        emb = embedding_matrix[rows]
        if hasattr(emb, "toarray"):
            emb = emb.toarray()
        return np.asarray(emb, dtype=float)

    def det_of(emb: np.ndarray) -> float:
        try:
            return covariance_det(emb, pca)
        except ValueError as exc:
            raise AnalysisError(
                f"PCA.transform 失败（嵌入维度 {emb.shape[1]} 与 PCA 基不符）。"
                f"fitted_pca.pkl 需配 sbert(384 维)；tfidf 请配合 --self-pca。原始错误：{exc}"
            ) from exc

    keys = sorted(identity_groups.keys(), key=str)
    if not keys:
        return None

    if len(keys) > 3 and triplet_iterations > 0:
        vendis: List[float] = []
        dets: List[float] = []
        mcs: List[float] = []
        n_used = 0
        for _ in range(triplet_iterations):
            chosen = rng.choice(len(keys), size=3, replace=False)
            members: List[Dict] = []
            for ci in chosen:
                members.extend(_capped(identity_groups[keys[int(ci)]], per_identity_cap))
            if len(members) < 2:
                continue
            emb = emb_of(members)
            vendis.append(vendi_from_distance_kernel(cosine_distance_matrix(emb)))
            dets.append(det_of(emb))
            mc = _unique_mcs(members)
            if not math.isnan(mc):
                mcs.append(mc)
            n_used = max(n_used, len(members))
        if not vendis:
            return None
        v = np.sort(np.asarray(vendis, dtype=float))
        d = np.sort(np.asarray(dets, dtype=float))
        stats = {
            "n_total": n_used,
            "vendi_point": float(np.mean(v)),
            "vendi_ci025": percentile_from_sorted(v, 2.5),
            "vendi_ci085": percentile_from_sorted(v, 8.5),
            "vendi_ci915": percentile_from_sorted(v, 91.5),
            "vendi_ci975": percentile_from_sorted(v, 97.5),
            "det_point": float(np.mean(d)),
            "det_ci025": percentile_from_sorted(d, 2.5),
            "det_ci085": percentile_from_sorted(d, 8.5),
            "det_ci915": percentile_from_sorted(d, 91.5),
            "det_ci975": percentile_from_sorted(d, 97.5),
            "mc_unique": float(np.mean(mcs)) if mcs else math.nan,
        }
    else:
        members = []
        for key in keys:
            members.extend(_capped(identity_groups[key], per_identity_cap))
        if len(keys) == 1 and per_identity_cap:


            members = members[: 3 * per_identity_cap]
        if len(members) < 2:
            return None
        emb = emb_of(members)
        vendi_point = vendi_from_distance_kernel(cosine_distance_matrix(emb))
        det_point = det_of(emb)
        vendi_ci025 = vendi_ci085 = vendi_ci915 = vendi_ci975 = math.nan
        det_ci025 = det_ci085 = det_ci915 = det_ci975 = math.nan
        if bootstrap > 0:
            vendis_b = np.empty(bootstrap, dtype=float)
            dets_b = np.empty(bootstrap, dtype=float)
            n = emb.shape[0]
            for b in range(bootstrap):
                idx = rng.integers(0, n, size=n)
                sample = emb[idx]
                vendis_b[b] = vendi_from_distance_kernel(cosine_distance_matrix(sample))
                dets_b[b] = det_of(sample)
            vendis_b = np.sort(vendis_b[~np.isnan(vendis_b)])
            dets_b = np.sort(dets_b[~np.isnan(dets_b)])
            if vendis_b.size:
                vendi_ci025 = percentile_from_sorted(vendis_b, 2.5)
                vendi_ci085 = percentile_from_sorted(vendis_b, 8.5)
                vendi_ci915 = percentile_from_sorted(vendis_b, 91.5)
                vendi_ci975 = percentile_from_sorted(vendis_b, 97.5)
            if dets_b.size:
                det_ci025 = percentile_from_sorted(dets_b, 2.5)
                det_ci085 = percentile_from_sorted(dets_b, 8.5)
                det_ci915 = percentile_from_sorted(dets_b, 91.5)
                det_ci975 = percentile_from_sorted(dets_b, 97.5)
        stats = {
            "n_total": len(members),
            "vendi_point": vendi_point,
            "vendi_ci025": vendi_ci025,
            "vendi_ci085": vendi_ci085,
            "vendi_ci915": vendi_ci915,
            "vendi_ci975": vendi_ci975,
            "det_point": det_point,
            "det_ci025": det_ci025,
            "det_ci085": det_ci085,
            "det_ci915": det_ci915,
            "det_ci975": det_ci975,
            "mc_unique": _unique_mcs(members),
        }

    row = dict(zip(AXIS_FIELDS, axis_key))
    row.update(stats)
    row.update(
        {
            "task_group": common.task_group(str(row.get("task_key") or "")),
            "coverage_scope": scope,
            "n_identities": len(keys),
            "embedding_backend": backend,
            "embedding_dim": embedding_matrix.shape[1],
            "det_pca": pca_label,
        }
    )
    return row


def sample_distances(
    embeddings_a: np.ndarray,
    sample_size: int,
    rng: np.random.Generator,
    embeddings_b: Optional[np.ndarray] = None,
) -> np.ndarray:
    if embeddings_b is None:
        dist = cosine_distance_matrix(embeddings_a)
        iu = np.triu_indices(dist.shape[0], k=1)
        pool = dist[iu]
    else:
        from sklearn.metrics.pairwise import cosine_distances

        pool = cosine_distances(embeddings_a, embeddings_b).ravel()
    if pool.size == 0:
        return np.array([], dtype=float)
    idx = rng.integers(0, pool.size, size=sample_size)
    return pool[idx]


def analyze_premise1_pair(
    axis_key: Tuple,
    identity_a: Tuple,
    items_a: Sequence[Dict],
    identity_b: Tuple,
    items_b: Sequence[Dict],
    embeddings_index: Dict[int, int],
    embedding_matrix: np.ndarray,
    rng: np.random.Generator,
) -> Optional[Dict]:
    from scipy.stats import ttest_ind

    rows_a = [embeddings_index[id(item)] for item in items_a]
    rows_b = [embeddings_index[id(item)] for item in items_b]
    emb_a = embedding_matrix[rows_a]
    emb_b = embedding_matrix[rows_b]
    if hasattr(emb_a, "toarray"):
        emb_a = emb_a.toarray()
    if hasattr(emb_b, "toarray"):
        emb_b = emb_b.toarray()
    emb_a = np.asarray(emb_a, dtype=float)
    emb_b = np.asarray(emb_b, dtype=float)
    if emb_a.shape[0] < 2 or emb_b.shape[0] < 2:
        return None

    within = np.concatenate(
        [sample_distances(emb_a, 500, rng), sample_distances(emb_b, 500, rng)]
    )
    across = sample_distances(emb_a, 1000, rng, embeddings_b=emb_b)
    if within.size == 0 or across.size == 0:
        return None


    t_stat, p_value = ttest_ind(across, within, equal_var=False, alternative="greater")


    ws, wp = _persample_premise(emb_a, emb_b)


    mcs_a = [it.get("mc") for it in items_a if _valid_mc(it.get("mc"))]
    mcs_b = [it.get("mc") for it in items_b if _valid_mc(it.get("mc"))]
    if mcs_a and mcs_b:
        chisq, chisq_p = _chisq_homogeneity([int(m) for m in mcs_a], [int(m) for m in mcs_b])
    else:
        chisq, chisq_p = math.nan, math.nan

    row = dict(zip(AXIS_FIELDS, axis_key))
    row.update(
        {
            "task_group": common.task_group(str(row.get("task_key") or "")),
            "identity_a": identity_a[1],
            "identity_b": identity_b[1],
            "n_a": emb_a.shape[0],
            "n_b": emb_b.shape[0],
            "mean_within": float(np.mean(within)),
            "mean_across": float(np.mean(across)),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "mean_within_persample": ws[0],
            "mean_across_persample": ws[1],
            "wilcoxon_stat": ws[2],
            "wilcoxon_p": wp,
            "chisq": chisq,
            "chisq_p": chisq_p,
        }
    )
    return row


def _persample_premise(emb_a: np.ndarray, emb_b: np.ndarray):
    from scipy.stats import wilcoxon
    from sklearn.metrics.pairwise import cosine_distances

    na, nb = emb_a.shape[0], emb_b.shape[0]
    d_aa = cosine_distances(emb_a)
    d_bb = cosine_distances(emb_b)
    d_ab = cosine_distances(emb_a, emb_b)

    within_per = np.concatenate([d_aa.sum(axis=1) / (na - 1), d_bb.sum(axis=1) / (nb - 1)])
    across_per = np.concatenate([d_ab.mean(axis=1), d_ab.mean(axis=0)])
    mean_within = float(np.mean(within_per))
    mean_across = float(np.mean(across_per))
    if np.any((across_per - within_per) != 0):
        stat, p = wilcoxon(across_per, within_per, alternative="greater")
        return (mean_within, mean_across, float(stat)), float(p)
    return (mean_within, mean_across, math.nan), math.nan


def write_csv(path: str, fields: Sequence[str], rows: Sequence[Dict]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Coverage(Vendi/det) 与 Premise 1(t 检验) 分析，纯 LLM、无需人类数据。"
    )
    parser.add_argument("--input", nargs="+", default=[], help="一个或多个生成结果 JSONL。")
    parser.add_argument("--legacy-pkl", default=None,
                        help="论文公开的旧模型聚合生成结果（llm_generations.pkl），与新模型同管线分析。")
    parser.add_argument("--coverage-csv", default="outputs/modern_coverage.csv")
    parser.add_argument("--premise1-csv", default="outputs/modern_premise1.csv")
    parser.add_argument("--embedding-backend", choices=["auto", "sbert", "tfidf"], default="auto")
    parser.add_argument("--sbert-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--pca-path", default="files/fitted_pca.pkl", help="原作者 384->10 维 PCA。")
    parser.add_argument("--self-pca", action="store_true",
                        help="忽略 pca-path，对本次全部回答自 fit 一次 PCA（所有分组共用同一基）。")
    parser.add_argument("--pca-components", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--min-responses", type=int, default=2, help="单身份最少样本数，少于则忽略。")
    parser.add_argument(
        "--per-identity-cap",
        type=int,
        default=None,
        help="Coverage 时每个身份最多取多少条（对齐论文设 33；默认全用）。",
    )
    parser.add_argument("--triplet-iterations", type=int, default=1000,
                        help="身份数>3 的分组随机抽 3 身份重复计算的次数（原作者 1000；0=直接全并）。")
    parser.add_argument("--no-identity-clean", action="store_true",
                        help="关闭身份词清洗。论文 Methods 对所有分析先移除显式身份词，默认开启。")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="只打印将分析的身份轴分组，不计算。")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = load_jsonl(args.input) if args.input else []
    if args.legacy_pkl:
        records.extend(common.load_legacy_generations(args.legacy_pkl))
    if not records:
        raise AnalysisError("没有可用记录（--input 的 JSONL 为空或全是空 response）。")
    records = expand_r2b_records(records)
    if not args.no_identity_clean:

        common.clean_records(records)

    axis_groups = group_by_axis(records)

    if args.dry_run:
        for axis_key, items in axis_groups.items():
            identity_groups = group_by_identity(items)
            preview = dict(zip(AXIS_FIELDS, axis_key))
            preview["n_identities"] = len(identity_groups)
            preview["n_total"] = len(items)
            print(json.dumps(preview, ensure_ascii=False))
        return

    embedding_matrix, backend = embed_all(records, args.embedding_backend, args.sbert_model, args.batch_size)
    embeddings_index = {id(item): i for i, item in enumerate(records)}
    rng = np.random.default_rng(args.seed)


    pca, pca_label = load_pca(args.pca_path, args.self_pca, embedding_matrix, args.pca_components)

    coverage_rows: List[Dict] = []
    premise1_rows: List[Dict] = []

    for axis_key, items in axis_groups.items():
        identity_groups = group_by_identity(items)

        identity_groups = {
            key: value for key, value in identity_groups.items() if len(value) >= args.min_responses
        }
        if not identity_groups:
            continue

        axis = axis_key[AXIS_FIELDS.index("identity_axis")]
        for scope, scope_groups in _coverage_scopes(str(axis), identity_groups):
            coverage = analyze_coverage(
                axis_key,
                scope,
                scope_groups,
                embeddings_index,
                embedding_matrix,
                backend,
                pca,
                pca_label,
                args.bootstrap,
                rng,
                args.per_identity_cap,
                args.triplet_iterations,
            )
            if coverage is not None:
                coverage_rows.append(coverage)

        for (identity_a, items_a), (identity_b, items_b) in combinations(identity_groups.items(), 2):
            pair = analyze_premise1_pair(
                axis_key,
                identity_a,
                items_a,
                identity_b,
                items_b,
                embeddings_index,
                embedding_matrix,
                rng,
            )
            if pair is not None:
                premise1_rows.append(pair)

    coverage_rows.sort(key=lambda r: tuple(str(r.get(f)) for f in AXIS_FIELDS + ["coverage_scope"]))
    premise1_rows.sort(key=lambda r: tuple(str(r.get(f)) for f in AXIS_FIELDS + ["identity_a", "identity_b"]))

    write_csv(args.coverage_csv, COVERAGE_FIELDS, coverage_rows)
    write_csv(args.premise1_csv, PREMISE1_FIELDS, premise1_rows)

    print(f"Embedding backend: {backend}")
    print(f"Coverage: 写出 {len(coverage_rows)} 行 -> {args.coverage_csv}")
    print(f"Premise 1: 写出 {len(premise1_rows)} 行（身份两两配对）-> {args.premise1_csv}")


if __name__ == "__main__":
    main()
