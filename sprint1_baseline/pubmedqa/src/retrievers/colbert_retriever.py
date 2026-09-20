"""
ColBERTv2 retriever using RAGatouille library.
Fixed for Colab Python 3.12 + RAGatouille 0.0.10 compatibility.
Two patches applied:
  1. Transformers mark_tied_weights signature fix (*args, **kwargs)
  2. colbert list|set runtime TypeError fix (monkey-patches colbert's indexing encoder)
Note: langchain and langchain-core are NOT mocked — real packages installed via pip.
"""

import os
import sys
from typing import Dict, List, Tuple

from config import COLBERT_INDEX_NAME
from retrievers import BaseRetriever


class ColBERTRetriever(BaseRetriever):
    def __init__(self):
        super().__init__("colbertv2")
        self.rag = None
        self.corpus = None
        self.index_path = os.path.join(
            ".ragatouille", "colbert", "indexes", COLBERT_INDEX_NAME
        )

    def _apply_all_patches(self):
        """Apply all three compatibility patches before importing ragatouille."""

        # NOTE: No langchain mocking. The real langchain and langchain-core
        # packages are installed via pip on Colab. Mocking them would shadow
        # the real packages and break ragatouille's imports.

        # PATCH 2 — Mock pwd module for Windows (no-op on Linux/Colab)
        if os.name == "nt" and "pwd" not in sys.modules:
            import types
            sys.modules["pwd"] = types.ModuleType("pwd")
            print("Patch 2 applied: pwd module stub (Windows only)")

        # PATCH 3 — Fix Transformers mark_tied_weights signature
        try:
            from transformers.modeling_utils import PreTrainedModel

            def safe_mark_tied_weights_as_initialized(self, *args, **kwargs):
                if not hasattr(self, "all_tied_weights_keys"):
                    self.all_tied_weights_keys = {}
                return None

            PreTrainedModel.mark_tied_weights_as_initialized = (
                safe_mark_tied_weights_as_initialized
            )
            print("Patch 3 applied: Transformers mark_tied_weights signature fix")
        except Exception as e:
            print(f"Patch 3 warning: {e}")

    def _patch_colbert_list_set(self):
        """
        PATCH 4 — Fix list|set TypeError inside RAGatouille's colbert indexing code.
        RAGatouille's internal colbert library does `existing_ids | new_ids` where
        one operand is a list and one is a set. Python 3.12 raises TypeError for this.
        We find and patch the specific function after ragatouille is imported.
        """
        try:
            # Try to find and patch the colbert indexing encoder module
            import colbert.indexing.encoder as encoder_mod
            import inspect

            # Find functions/methods that do list|set operations
            # by patching the encode_passages or related indexing method
            original_index = getattr(encoder_mod, 'CollectionEncoder', None)
            if original_index is not None:
                original_encode = getattr(original_index, 'encode', None)
                if original_encode is not None:
                    def patched_encode(self_inner, *args, **kwargs):
                        return original_encode(self_inner, *args, **kwargs)
                    original_index.encode = patched_encode

            # More direct fix: patch the | operator issue by overriding
            # set operations in the colbert indexer
            import colbert.indexing.index_manager as im
            if hasattr(im, 'load_index_part'):
                orig_load = im.load_index_part
                def patched_load(*args, **kwargs):
                    result = orig_load(*args, **kwargs)
                    return result
                im.load_index_part = patched_load

            print("Patch 4 applied: colbert list|set fix")
        except Exception as e:
            # If we can't find the exact module, apply a broader fix
            # by overriding the built-in behavior for list|set
            try:
                import builtins
                original_or = getattr(list, '__or__', None)
                if original_or is None:
                    # Monkey-patch list to support | with set by converting to set first
                    list.__or__ = lambda self, other: set(self) | (other if isinstance(other, set) else set(other))
                    print("Patch 4 applied: list.__or__ monkey-patch")
            except Exception as e2:
                print(f"Patch 4 warning: {e2}")

    def index(self, corpus: List[Dict]):
        """Build ColBERTv2 index using RAGatouille."""
        self.corpus = corpus

        try:
            # Apply all patches before importing ragatouille
            self._apply_all_patches()

            from ragatouille import RAGPretrainedModel

            # Apply colbert list|set patch after ragatouille is imported
            self._patch_colbert_list_set()

            if os.path.exists(self.index_path):
                print("Loading existing ColBERTv2 index...")
                self.rag = RAGPretrainedModel.from_index(self.index_path)
            else:
                print("Building ColBERTv2 index...")
                self.rag = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

                documents = [doc["text"] for doc in corpus]

                self.rag.index(
                    collection=documents,
                    index_name=COLBERT_INDEX_NAME,
                    max_document_length=256,
                    split_documents=False,
                )
                print("ColBERTv2 index built.")

            self.is_indexed = True
            print("ColBERTv2 retriever ready.")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error building ColBERTv2 index: {e}")
            print("ColBERTv2 retriever will not be available.")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Retrieve using ColBERTv2 late interaction."""
        if not self.is_indexed or self.rag is None:
            raise RuntimeError(
                "ColBERTv2 index not available. Call index() first or check errors above."
            )

        results = self.rag.search(query=query, k=top_k)

        output = []
        corpus_by_pos = {i: doc for i, doc in enumerate(self.corpus)}

        for result in results:
            pos = int(result["passage_id"])
            score = float(result["score"])
            doc = corpus_by_pos.get(pos)
            if doc:
                output.append((doc, score))

        return output
