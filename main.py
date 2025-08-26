#!/usr/bin/env python3
"""
Similar paragraphs finder for .docx files.

Features
- Batched embeddings with switchable backend (FlagEmbedding BGE or sentence-transformers)
- Optional pre-embedding de-duplication: off / exact / normalized
- Length-based masking for candidate pairs (min ratio and/or max abs diff)
- Chunked cosine similarities with unique (i<j) pairs
- TXT / CSV / HTML report (with simple inline highlight of common spans)
- Embedding cache on disk per (file, mtime, backend, model, min_len, max_length)

Install deps:
  pip install python-docx FlagEmbedding sentence-transformers numpy tqdm

Usage examples:
  python similar_paragraphs_final.py \
    --input kk2.docx --threshold 0.9 --topk 5 --html report.html

  # normalized de-duplication and length mask
  python similar_paragraphs_final.py \
    --input kk2.docx --dedupe normalized \
    --len-ratio-min 0.7 --len-diff-max 300 --csv pairs.csv
"""
from __future__ import annotations

import argparse
import html as _html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from docx import Document
from tqdm import tqdm

from embeddings_backends import make_backend

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
    """Lenient normalization used to collapse duplicates.
    - lowercase, remove punctuation/symbols, collapse whitespace
    """
    try:
        import regex as _regex  # supports \p{P} / \p{S}
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
    backend_kind: str
    model_name: str
    min_len: int
    max_length: int

    def digest(self) -> str:
        blob = json.dumps(
            {
                "src_path": self.src_path,
                "mtime_ns": self.mtime_ns,
                "backend": self.backend_kind,
                "model": self.model_name,
                "min_len": self.min_len,
                "max_length": self.max_length,
            },
            sort_keys=True,
        ).encode("utf-8")
        import hashlib
        return hashlib.sha1(blob).hexdigest()

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
    lengths: Optional[np.ndarray] = None,
    len_ratio_min: float = 0.6,
    len_diff_max: int = 0,
) -> List[Tuple[int, int, float]]:
    """Return unique (i, j, sim) with i < j.

    Length mask:
      - ratio:  min(len_i, len_j) / max(len_i, len_j) >= len_ratio_min
      - diff:   |len_i - len_j| <= len_diff_max   (0 disables)
    """
    N = E.shape[0]
    if lengths is None:
        lengths = np.ones(N, dtype=np.int32)
    else:
        lengths = lengths.astype(np.int32, copy=False)

    results: List[Tuple[int, int, float]] = []
    for start in tqdm(range(0, N, chunk_size), desc="similarity", unit="chunk"):
        stop = min(start + chunk_size, N)
        S = E[start:stop] @ E.T  # (chunk, N)

        # Build length mask for this chunk vs all
        la = lengths[start:stop].astype(np.float32)[:, None]
        lb = lengths.astype(np.float32)[None, :]
        minlen = np.minimum(la, lb)
        maxlen = np.maximum(la, lb)
        ratio_mask = (minlen / np.maximum(maxlen, 1.0)) >= float(len_ratio_min)
        if len_diff_max and len_diff_max > 0:
            diff_mask = (np.abs(la - lb) <= int(len_diff_max))
            mask = ratio_mask & diff_mask
        else:
            mask = ratio_mask

        # mask out invalid comps and self-sim
        S[~mask] = -1.0
        rows = stop - start
        for r in range(rows):
            S[r, start + r] = -1.0

        # top-k per row
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

    # dedup pairs, keep max score
    dedup: Dict[Tuple[int, int], float] = {}
    for a, b, s in results:
        key = (a, b)
        if key not in dedup or s > dedup[key]:
            dedup[key] = s
    out = [(a, b, s) for (a, b), s in dedup.items()]
    out.sort(key=lambda t: t[2], reverse=True)
    return out

# ---------------------------
# De-duplication
# ---------------------------

def dedupe_texts(texts: List[str], mode: str) -> Tuple[List[str], Dict[int, int], Dict[int, List[int]]]:
    """
    Returns (kept_texts, old_to_new_idx, groups)
    groups[new_idx] = list of original indices collapsed into it (incl. representative).
    Modes: 'off' | 'exact' | 'normalized'
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
) -> str:
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
) -> None:
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
    a_out: List[str] = []
    b_out: List[str] = []
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
) -> None:
    def color_for_score(s: float) -> str:
        # 0.0 -> light gray, 1.0 -> deeper green
        g = int(240 - min(max((s - 0.7) / 0.3, 0.0), 1.0) * 160)  # focus 0.7..1.0
        return f"rgb(200,{g},200)"

    rows: List[str] = []
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
        description="Find similar paragraphs in a DOCX using embeddings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", required=True, help="Path to .docx file")
    p.add_argument("--backend", choices=["bge", "st"], default="bge")
    p.add_argument("--model", default="BAAI/bge-m3", help="Embedding model name")
    p.add_argument("--device", default="auto", help="Device for model (auto|cpu|cuda:0)")
    p.add_argument("--no-fp16", action="store_true", help="Disable FP16 (BGE backend)")
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote code for sentence-transformers models",
    )
    p.add_argument("--batch-size", type=int, default=32, help="Embedding batch size")
    p.add_argument("--min-len", type=int, default=25, help="Minimum paragraph length to keep")
    p.add_argument("--max-length", type=int, default=0, help="Override model max_length (0=auto)")
    p.add_argument("--threshold", type=float, default=0.88, help="Cosine similarity threshold")
    p.add_argument("--topk", type=int, default=5, help="Candidates per paragraph before thresholding")
    p.add_argument("--chunk-size", type=int, default=2048, help="Chunk size for similarity matmul")
    p.add_argument("--len-ratio-min", type=float, default=0.6, help="Skip pairs where shorter/longer < this ratio (0..1)")
    p.add_argument("--len-diff-max", type=int, default=0, help="Skip pairs whose abs length difference exceeds this many characters; 0=disabled")
    p.add_argument("--output", type=Path, default=Path("similar.txt"), help="Write human-readable TXT here")
    p.add_argument("--csv", type=Path, default=None, help="Optional CSV path for results")
    p.add_argument("--html", type=Path, default=None, help="Optional HTML report path")
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache_embeddings"),
        help="Directory for embedding cache",
    )
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

    # Choose max length
    longest = max(len(t) for t in texts)
    max_len = args.max_length if args.max_length > 0 else choose_max_length(longest)

    # Cache key and path
    key = CacheKey(
        src_path=str(src.resolve()),
        mtime_ns=int(src.stat().st_mtime_ns),
        backend_kind=args.backend,
        model_name=args.model,
        min_len=args.min_len,
        max_length=max_len,
    )
    cache_dir: Path = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    base = cache_dir / key.digest()
    emb_path = base.with_suffix(".embeddings.npy")
    meta_path = base.with_suffix(".meta.json")

    embs: Optional[np.ndarray] = None
    if emb_path.is_file() and meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("backend") == args.backend and meta.get("model") == args.model:
                embs = np.load(emb_path)
                if (
                    embs.shape == tuple(meta.get("shape", ()))
                    and str(embs.dtype) == meta.get("dtype")
                    and embs.shape[0] == len(texts)
                ):
                    print(
                        f"[cache] loaded embeddings {embs.shape} from {emb_path.name}"
                    )
                else:
                    embs = None
        except Exception:
            embs = None

    if embs is None:
        backend = make_backend(
            kind=args.backend,
            model_name=args.model,
            device=args.device,
            fp16=not args.no_fp16,
            trust_remote_code=args.trust_remote_code,
        )
        print(
            f"[embed] encoding {len(texts)} paragraphs (batch={args.batch_size}, max_length={max_len}) …"
        )
        embs = backend.encode(texts, batch_size=args.batch_size, max_length=max_len)
        embs = np.asarray(embs, dtype=np.float32, order="C")
        np.save(emb_path, embs)
        meta = {
            "backend": args.backend,
            "model": args.model,
            "shape": list(embs.shape),
            "dtype": str(embs.dtype),
        }
        meta_path.write_text(json.dumps(meta))

    print(
        f"[backend] backend={args.backend} model={args.model} device={args.device} dim={embs.shape[1]}"
    )

    # Normalize & compute
    E = l2_normalize(embs.astype(np.float32, copy=False))
    lengths = np.array([len(t) for t in texts], dtype=np.int32)
    print(
        f"[sim] computing similarities (topk={args.topk}, thr={args.threshold}, chunk={args.chunk_size}, "
        f"len_ratio_min={args.len_ratio_min}, len_diff_max={args.len_diff_max}) …"
    )
    pairs = topk_similarities(
        E,
        topk=args.topk,
        threshold=args.threshold,
        chunk_size=args.chunk_size,
        lengths=lengths,
        len_ratio_min=args.len_ratio_min,
        len_diff_max=args.len_diff_max,
    )

    # Output
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
