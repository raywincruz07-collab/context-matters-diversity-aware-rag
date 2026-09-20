"""
Dense retriever implementations.

Important:
- Original DPR is implemented separately in retrievers/dpr_original_retriever.py.
- This file keeps ContrieverRetriever.
- The previous MiniLM-based DPR substitute has been disabled and must not be used as DPR.
- Contriever uses facebook/contriever directly.
- No MiniLM fallback is allowed for Contriever, because that would make results mislabeled.
"""

import os
import pickle
from typing import Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import EMBEDDINGS_DIR, INDEX_DIR
from retrievers import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Generic dense retriever with FAISS backend."""

    def __init__(self, name: str, model_name: str):
        super().__init__(name)
        self.model_name = model_name
        self.model = None
        self.faiss_index = None
        self.embedding_dim = None

    def _load_model(self):
        if self.model is None:
            print(f"Loading embedding model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def _encode(
        self, texts: List[str], batch_size: int = 32, show_progress: bool = True
    ) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,  # for cosine similarity via dot product
        )
        return embeddings.astype(np.float32)

    def index(self, corpus: List[Dict]):
        """Build FAISS index from corpus."""
        self.corpus = corpus
        emb_path = os.path.join(EMBEDDINGS_DIR, f"{self.name}_embeddings.npy")
        idx_path = os.path.join(INDEX_DIR, f"{self.name}_faiss.index")

        # Load or compute embeddings
        if os.path.exists(emb_path):
            print(f"Loading {self.name} embeddings from disk...")
            embeddings = np.load(emb_path)
        else:
            print(f"Computing {self.name} embeddings (this may take a while on CPU)...")
            texts = [doc["text"] for doc in corpus]
            embeddings = self._encode(texts, batch_size=16)
            np.save(emb_path, embeddings)
            print(f"Saved embeddings to {emb_path}")

        self.embedding_dim = embeddings.shape[1]

        # Build or load FAISS index
        if os.path.exists(idx_path):
            print(f"Loading {self.name} FAISS index...")
            self.faiss_index = faiss.read_index(idx_path)
        else:
            print(f"Building {self.name} FAISS index...")
            self.faiss_index = faiss.IndexFlatIP(
                self.embedding_dim
            )  # Inner product (cosine since normalized)
            self.faiss_index.add(embeddings)
            faiss.write_index(self.faiss_index, idx_path)
            print(f"Saved FAISS index to {idx_path}")

        self.is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using dense embeddings + FAISS."""
        if not self.is_indexed:
            raise RuntimeError("Index not built. Call index() first.")

        query_emb = self._encode([query], show_progress=False)
        scores, indices = self.faiss_index.search(query_emb, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # FAISS returns -1 for missing results
                results.append((self.corpus[idx], float(score)))

        return results

    def unload_model(self):
        """Free the embedding model from memory."""
        self.model = None
        import gc

        gc.collect()


class DPRRetriever(DenseRetriever):
    def __init__(self):
        raise RuntimeError(
            "Deprecated fake DPR removed. Use OriginalDPRRetriever from retrievers/dpr_original_retriever.py instead."
        )


class ContrieverRetriever(DenseRetriever):
    """
    Contriever retriever using facebook/contriever.

    If the model cannot load, the retriever fails loudly instead of falling back to another model.
    This preserves experimental integrity.
    """

    def __init__(self):
        super().__init__(name="contriever", model_name="facebook/contriever")

    def _load_model(self):
        """Load Contriever with fallback."""
        if self.model is None:
            try:
                print(f"Loading Contriever model...")
                import torch
                from transformers import AutoModel, AutoTokenizer

                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.base_model = AutoModel.from_pretrained(self.model_name)
                self.base_model.eval()

                # Wrap in a callable that mimics sentence-transformers interface
                class ContrieverWrapper:
                    def __init__(self, model, tokenizer):
                        self.model = model
                        self.tokenizer = tokenizer

                    def get_sentence_embedding_dimension(self):
                        return 768

                    def encode(
                        self,
                        texts,
                        batch_size=32,
                        show_progress_bar=True,
                        normalize_embeddings=True,
                    ):
                        all_embeddings = []
                        iterator = range(0, len(texts), batch_size)
                        if show_progress_bar:
                            from tqdm import tqdm

                            iterator = tqdm(iterator, desc="Encoding")

                        for i in iterator:
                            batch = texts[i : i + batch_size]
                            inputs = self.tokenizer(
                                batch,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt",
                            )
                            with torch.no_grad():
                                outputs = self.model(**inputs)
                            # Mean pooling
                            mask = inputs["attention_mask"].unsqueeze(-1).float()
                            embeddings = (outputs.last_hidden_state * mask).sum(
                                1
                            ) / mask.sum(1)

                            if normalize_embeddings:
                                embeddings = torch.nn.functional.normalize(
                                    embeddings, p=2, dim=1
                                )

                            all_embeddings.append(embeddings.numpy())

                        return np.concatenate(all_embeddings, axis=0)

                self.model = ContrieverWrapper(self.base_model, self.tokenizer)
                self.embedding_dim = 768
                print("Contriever loaded successfully.")

            except Exception as e:
                raise RuntimeError(
                    "Failed to load facebook/contriever. "
                    "Contriever fallback to MiniLM is disabled to preserve experimental integrity. "
                    "Fix the environment/model download instead of substituting another model."
                ) from e
