from __future__ import annotations

from typing import List
import numpy as np


class EmbeddingBackend:
    """Abstract interface for embedding backends."""

    name: str

    def encode(self, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        """Return a matrix of shape [N, D] as np.float32 without normalization."""
        raise NotImplementedError


class FlagBGEBackend(EmbeddingBackend):
    def __init__(self, model_name: str, device: str = "auto", fp16: bool = True):
        try:
            from FlagEmbedding import BGEM3FlagModel
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Please `pip install FlagEmbedding` to use the BGE backend"
            ) from e

        self.name = f"flag:{model_name}"
        self.model = BGEM3FlagModel(model_name, use_fp16=fp16)

    def encode(self, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        out = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )["dense_vecs"]
        return np.asarray(out, dtype=np.float32, order="C")


class STBackend(EmbeddingBackend):
    def __init__(self, model_name: str, device: str = "auto", trust_remote_code: bool = False):
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Please `pip install sentence-transformers` to use this backend"
            ) from e

        if device == "auto":
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(
            model_name, device=device, trust_remote_code=trust_remote_code
        )
        self.name = f"st:{model_name}"

    def encode(self, texts: List[str], batch_size: int, max_length: int) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        return embs.astype(np.float32, copy=False)


def make_backend(
    kind: str,
    model_name: str,
    device: str = "auto",
    fp16: bool = True,
    trust_remote_code: bool = False,
) -> EmbeddingBackend:
    kind = kind.lower()
    if kind == "bge":
        return FlagBGEBackend(model_name=model_name, device=device, fp16=fp16)
    if kind == "st":
        return STBackend(
            model_name=model_name, device=device, trust_remote_code=trust_remote_code
        )
    raise ValueError(f"Unknown backend kind: {kind}")
