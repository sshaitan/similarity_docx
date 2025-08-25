#!/usr/bin/env python3
"""
Find near-duplicate / highly similar paragraphs in a .docx document.

New in this version:
- Pre-embedding de-duplication with configurable modes (off / exact / normalized)
- Simple HTML report with inline similarity coloring and diff-like highlighting
- Same speedups as before: batched embeddings, caching, chunked cosine

Dependencies: python-docx, FlagEmbedding, numpy, tqdm
Optional (HTML diff quality): beautifulsoup4 (only for escaping convenience; we fallback if missing)

Examples:
  python refactored_similar_paragraphs.py \
    --input kk2.docx --threshold 0.9 --topk 5 --html report.html

  # strict dedupe of exact duplicates before encoding
  python refactored_similar_paragraphs.py --input kk2.docx --dedupe exact

  # more aggressive dedupe on whitespace/case/punct normalization
  python refactored_similar_paragraphs.py --input kk2.docx --dedupe normalized
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from docx import Document
from tqdm import tqdm

try:
    from FlagEmbedding import BGEM3FlagModel
except Exception as e:  # pragma: no cover
    print("[!] Please `pip install FlagEmbedding` (and python-docx, tqdm, numpy).", file=sys.stderr)
    raise

# ---------------------------
# Utilities
# ---------------------------

def read_paragraphs(docx_path: Path, min_len: int = 25) -> List[str]:
    if not docx_path.is_file():
        raise FileNotFoundError(f"No such file: {docx_path}")
    doc = Document(str(docx_path))
    out: List[str] = []
    for p in doc.paragraphs:
        s = (p.text or "").strip().replace("\u00A0", " ")
        if len(s) >= min_len:
            out.append(s)
    return out

_whitespace_re = re.compile(r"\s+", re.UNICODE)


def normalize_text(s: str) -> str:
    """Lenient normalization for duplicate collapsing.
    - lowercase
    - collapse whitespace
    - strip punctuation/symbols
    """
    try:
        import regex as _regex  # better \p{P} / \p{S}
        punct = _regex.sub(r"[\p{P}\p{S}]", " ", s)
    except Exception:
        punct = re.sub(r"[\W_]", " ", s, flags=re.UNICODE)
    s2 = punct.lower()
    s2 = _whitespace_re.sub(" ", s2).strip()
    return s2


@dataclass(frozen=True)
class CacheKey:
    src_path: str
    mtime_ns: int
    model_name: str
    min_len: int
    max_length: int

    def digest(self) -> str:
        blob = json.dumps(
            {
                "src_path": self.src_path,
                "mtime_ns": self.mtime_ns,
                "model": self.model_name,
                "min_len": self.min_len,
                "max_length": self.max_length,
            },
            sort_keys=True,
        ).encode("utf-8")
        import hashlib
        return hashlib.sha1(blob).hexdigest()

# ---------------------------
# Embedding
# ---------------------------

def load_model(model_name: str, use_fp16: bool = True) -> BGEM3FlagModel:
    return BGEM3FlagModel(model_name, use_fp16=use_fp16)


def make_embeddings(
    texts: List[str],
    model: BGEM3FlagModel,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    embs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )["dense_vecs"]
    embs = np.asarray(embs)
    if embs.dtype != np.float32:
        embs = embs.astype(np.float32, copy=False)
    return embs

# ---------------------------
# Similarities
# ---------------------------

def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def topk_similarities(
    E: np.ndarray,
    topk: int,
    threshold: float,
    chunk_size: int = 2048,
) -> List[Tuple[int, int, float]]:
    N = E.shape[0]
    results: List[Tuple[int, int, float]] = []
    for start in tqdm(range(0, N, chunk_size), desc="similarity", unit="chunk"):
        stop = min(start + chunk_size, N)
        S = E[start:stop] @ E.T  # (chunk, N)
        rows = stop - start
        for r in range(rows):
            S[r, start + r] = -1.0
        if topk > 0 and topk < N:
            idx = np.argpartition(S, -topk, axis=1)[:, -topk:]
        else:
            idx = np.argsort(S, axis=1)[:, ::-1]
        for r in range(rows):
            row_scores = S[r, idx[r]]
            row_idx = idx[r]
            for j, s in zip(row_idx, row_scores):
                if s >= threshold:
                    i_abs = start + r
                    a, b = (i_abs, int(j)) if i_abs < j else (int(j), i_abs)
                    if a != b:
                        results.append((a, b, float(s)))
    dedup = {}
    for a, b, s in results:
        key = (a, b)
        if key not in dedup or s > dedup[key]:
            dedup[key] = s
    out = [(a, b, s) for (a, b), s in dedup.items()]
    out.sort(key=lambda t: t[2], reverse=True)
    return out

# ---------------------------
# Dedupe
# ---------------------------

def dedupe_texts(texts: List[str], mode: str) -> Tuple[List[str], Dict[int, int], Dict[int, List[int]]]:
    """
    Returns:
      kept_texts, old_to_new_idx, groups
    where groups[new_idx] = list of original indices collapsed into it (including the representative).
    Modes:
      - 'off': no dedupe
      - 'exact': collapse exact string duplicates
      - 'normalized': collapse by normalize_text(s)
    """
    if mode == "off":
        return texts, {i: i for i in range(len(texts))}, {i: [i] for i in range(len(texts))}

    key_map: Dict[str, int] = {}
    kept: List[str] = []
    old_to_new: Dict[int, int] = {}
    groups: Dict[int, List[int]] = {}

    for i, t in enumerate(texts):
        k = t if mode == "exact" else normalize_text(t)
        if k in key_map:
            rep = key_map[k]
            old_to_new[i] = rep
            groups.setdefault(rep, []).append(i)
        else:
            rep = len(kept)
            key_map[k] = rep
            kept.append(t)
            old_to_new[i] = rep
            groups[rep] = [i]
    return kept, old_to_new, groups

# ---------------------------
# Output (TXT/CSV/HTML)
# ---------------------------

def write_txt(
    pairs: List[Tuple[int, int, float]],
    texts: List[str],
    path: Optional[Path],
):
    lines: List[str] = []
    for i, j, s in pairs:
        lines.append(
            (
                "Параграф A (#%d):\n\n%s\n\n"
                "очень похож на Параграф B (#%d):\n\n%s\n\n"
                "с косинусным сходством = %.4f\n\n"
                "=============================================\n\n"
            )
            % (i, texts[i], j, texts[j], s)
        )
    text = "".join(lines)
    if path:
        path.write_text(text, encoding="utf-8")
    return text


def write_csv(
    pairs: List[Tuple[int, int, float]],
    texts: List[str],
    path: Path,
):
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "similarity", "text_i", "text_j"])
        for i, j, s in pairs:
            w.writerow([i, j, f"{s:.6f}", texts[i], texts[j]])


def _escape_html(s: str) -> str:
    return _html.escape(s, quote=True)


def _highlight_common(a: str, b: str) -> Tuple[str, str]:
    """Very lightweight diff-like highlight: mark common chunks (len>=8) with <mark>.
    Returns (html_a, html_b).
    """
    import difflib
    sm = difflib.SequenceMatcher(a=a, b=b)
    a_out = []
    b_out = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        a_seg = _escape_html(a[i1:i2])
        b_seg = _escape_html(b[j1:j2])
        if tag == 'equal' and (i2 - i1) >= 8 and (j2 - j1) >= 8:
            a_out.append(f"<mark>{a_seg}</mark>")
            b_out.append(f"<mark>{b_seg}</mark>")
        else:
            a_out.append(a_seg)
            b_out.append(b_seg)
    return "".join(a_out), "".join(b_out)


def write_html(
    pairs: List[Tuple[int, int, float]],
    texts: List[str],
    path: Path,
    dedupe_groups: Optional[Dict[int, List[int]]] = None,
):
    def color_for_score(s: float) -> str:
        # 0.0 -> light gray, 1.0 -> deeper green
        g = int(240 - min(max((s - 0.7) / 0.3, 0.0), 1.0) * 160)  # focus 0.7..1.0
        return f"rgb(200,{g},200)"

    rows = []
    for i, j, s in pairs:
        a_html, b_html = _highlight_common(texts[i], texts[j])
        dedupe_info_a = f"<div class='small'>Collapsed originals: {dedupe_groups.get(i, [i])}</div>" if dedupe_groups else ""
        dedupe_info_b = f"<div class='small'>Collapsed originals: {dedupe_groups.get(j, [j])}</div>" if dedupe_groups else ""
        rows.append(
            f"""
            <details style='background:{color_for_score(s)}; padding:10px; border-radius:12px; margin:10px 0'>
              <summary><b>#{i}</b> ↔ <b>#{j}</b> &nbsp; score=<code>{s:.4f}</code></summary>
              <div class='pair'>
                <div class='col'><h4>A (#{i})</h4><div class='para'>{a_html}</div>{dedupe_info_a}</div>
                <div class='col'><h4>B (#{j})</h4><div class='para'>{b_html}</div>{dedupe_info_b}</div>
              </div>
            </details>
            """
        )

    html = f"""
<!doctype html>
<meta charset='utf-8'>
<title>Similar Paragraphs Report</title>
<style>
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }}
 .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
 .para {{ white-space: pre-wrap; line-height: 1.4; }}
 mark {{ background: #fff59d; padding: 0 2px; border-radius: 3px; }}
 .small {{ color: #666; font-size: 12px; margin-top: 6px; }}
 .pill {{ display:inline-block; padding: 2px 8px; border:1px solid #ccc; border-radius:999px; margin-right:8px; }}
 .meta {{ margin-bottom: 12px; }}
</style>
<h1>Similar Paragraphs</h1>
<div class='meta'>
 <span class='pill'>pairs: {len(pairs)}</span>
</div>
{''.join(rows)}
"""
    path.write_text(html, encoding='utf-8')

# ---------------------------
# Main
# ---------------------------

def choose_max_length(longest_char_len: int) -> int:
    buckets = [512, 1024, 2048, 4096, 8192]
    for b in buckets:
        if longest_char_len <= b:
            return b
    return buckets[-1]


def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Find similar paragraphs in a DOCX using BGE-M3 embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to .docx file")
    p.add_argument("--model", default="BAAI/bge-m3", help="Model name for FlagEmbedding")
    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p.add_argument("--min-len", type=int, default=25, help="Minimum paragraph length to keep")
    p.add_argument("--max-length", type=int, default=0, help="Override model max_length (0=auto)")
    p.add_argument("--fp16", action="store_true", help="Use FP16 in model")
    p.add_argument("--threshold", type=float, default=0.88, help="Cosine similarity threshold")
    p.add_argument("--topk", type=int, default=5, help="Candidates per paragraph before thresholding")
    p.add_argument("--chunk-size", type=int, default=2048, help="Chunk size for similarity matmul")
    p.add_argument("--output", type=Path, default=Path("similar.txt"), help="Write human-readable TXT here")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV path for results")
    p.add_argument("--html", type=Path, default=None, help="Optional HTML report path")
    p.add_argument("--cache-dir", type=Path, default=Path(".cache_bge"), help="Directory for embedding cache")
    p.add_argument("--dedupe", choices=["off", "exact", "normalized"], default="normalized", help="Collapse obvious duplicates before embedding")

    args = p.parse_args(list(argv) if argv is not None else None)

    src = Path(args.input)
    texts_raw = read_paragraphs(src, min_len=args.min_len)
    if not texts_raw:
        print("[!] No paragraphs after filtering. Check --min-len or document content.")
        return 1

    # Dedupe (pre-embedding)
    texts, old_to_new, groups = dedupe_texts(texts_raw, mode=args.dedupe)
    if args.dedupe != "off":
        print(f"[dedupe] {len(texts_raw)} -> {len(texts)} unique (mode={args.dedupe})")

    longest = max(len(t) for t in texts)
    max_len = args.max_length if args.max_length > 0 else choose_max_length(longest)

    key = CacheKey(
        src_path=str(src.resolve()),
        mtime_ns=int(src.stat().st_mtime_ns),
        model_name=args.model,
        min_len=args.min_len,
        max_length=max_len,
    )

    # Cache
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb_path = cache_dir / f"{key.digest()}.embeddings.npy"
    if emb_path.is_file():
        embs = np.load(emb_path)
        print(f"[cache] loaded embeddings {embs.shape} from {emb_path.name}")
    else:
        print(f"[model] loading {args.model} (fp16={args.fp16}) …")
        model = load_model(args.model, use_fp16=args.fp16)
        print(f"[embed] encoding {len(texts)} paragraphs (batch={args.batch_size}, max_length={max_len}) …")
        embs = make_embeddings(texts, model, args.batch_size, max_length=max_len)
        np.save(emb_path, embs)

    E = l2_normalize(embs.astype(np.float32, copy=False))

    print(f"[sim] computing similarities (topk={args.topk}, thr={args.threshold}, chunk={args.chunk_size}) …")
    pairs = topk_similarities(E, topk=args.topk, threshold=args.threshold, chunk_size=args.chunk_size)

    print(f"[out] {len(pairs)} pairs ≥ {args.threshold}")
    if args.output:
        write_txt(pairs, texts, args.output)
        print(f"[out] wrote TXT: {args.output}")
    if args.csv:
        write_csv(pairs, texts, args.csv)
        print(f"[out] wrote CSV: {args.csv}")
    if args.html:
        write_html(pairs, texts, args.html, dedupe_groups=groups)
        print(f"[out] wrote HTML: {args.html}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
