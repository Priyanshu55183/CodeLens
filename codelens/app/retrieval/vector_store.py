"""
ChromaDB vector store for code chunks.

Design decisions:
- We use ChromaDB's persistent client so embeddings survive restarts.
  Re-embedding the entire repo on every restart would be slow and costly.
- Each repo gets its own ChromaDB collection, named by "owner__repo" to
  avoid collisions. We use double underscore because ChromaDB collection
  names can't contain slashes.
- We store chunk metadata (file_path, start_line, end_line, language) in
  ChromaDB's metadata field so we can return citations without a second lookup.
- Embedding function: We use ChromaDB's built-in OpenAI embedding function
  for Phase 1. This will be swapped to a configurable function in later phases.
- We upsert (not insert) so re-indexing the same file is idempotent.

Interview note: ChromaDB uses HNSW (Hierarchical Navigable Small World)
under the hood — an approximate nearest neighbor algorithm. It's not exact,
but it's fast and good enough for retrieval (we fix mistakes with reranking
in Phase 3).
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from codelens.app.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk returned by vector search, with its similarity score.
    
    This is separate from Chunk because it adds the distance/score
    from the search, which the original Chunk doesn't have.
    """

    chunk_id: str
    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    distance: float          # Lower = more similar (cosine distance in ChromaDB)
    symbol_name: str = ""
    symbol_type: str = ""
    parent_class: str = ""
    imports: str = ""

    @property
    def citation(self) -> str:
        """Format as a citation string: [file_path:start-end]"""
        if self.symbol_name:
            symbol = f" ({self.parent_class}.{self.symbol_name})" if self.parent_class else f" ({self.symbol_name})"
            return f"[{self.file_path}:{self.start_line}-{self.end_line}{symbol}]"
        return f"[{self.file_path}:{self.start_line}-{self.end_line}]"


class VectorStore:
    """Manages ChromaDB collections for repo code chunks.

    Usage:
        store = VectorStore(chroma_dir, api_key="sk-...")
        store.add_chunks("owner", "repo", chunks)
        results = store.query("owner", "repo", "how does auth work?", top_k=5)
    """

    def __init__(self, chroma_dir: Path, api_key: str, embedding_model: str = "text-embedding-3-small"):
        """Initialize the vector store.

        Args:
            chroma_dir: Directory where ChromaDB persists data.
            api_key: OpenAI API key for the embedding function.
            embedding_model: Name of the OpenAI embedding model.
        """
        self.chroma_dir = chroma_dir
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(chroma_dir))
        self.embedding_fn = OpenAIEmbeddingFunction(
            api_key=api_key,
            model_name=embedding_model,
        )
        logger.info("ChromaDB initialized at %s", chroma_dir)

    def _collection_name(self, owner: str, repo: str) -> str:
        """Build a ChromaDB-safe collection name.
        
        ChromaDB names must be 3-63 chars, alphanumeric with underscores/hyphens.
        We use "owner__repo" format.
        """
        name = f"{owner}__{repo}".lower()
        # Sanitize: keep only alphanumeric, underscore, hyphen
        name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in name)
        # Ensure length constraints
        if len(name) < 3:
            name = name + "___"
        return name[:63]

    def get_or_create_collection(self, owner: str, repo: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection for a repo."""
        name = self._collection_name(owner, repo)
        collection = self.client.get_or_create_collection(
            name=name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"},  # Use cosine similarity
        )
        logger.info("Collection '%s' has %d documents", name, collection.count())
        return collection

    def add_chunks(self, owner: str, repo: str, chunks: list[Chunk], batch_size: int = 100) -> int:
        """Add chunks to the vector store.

        Uses upsert so re-indexing is idempotent (same chunk_id = overwrite).

        Args:
            owner: GitHub repo owner.
            repo: GitHub repo name.
            chunks: List of Chunk objects to store.
            batch_size: Number of chunks to upsert at once (ChromaDB limit).

        Returns:
            Number of chunks added.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return 0

        collection = self.get_or_create_collection(owner, repo)

        # Batch the upserts (ChromaDB has a limit per call)
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.content for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            logger.debug("Upserted batch %d-%d", i, i + len(batch))

        total = collection.count()
        logger.info("Added %d chunks. Collection now has %d documents.", len(chunks), total)
        return len(chunks)

    def query(
        self,
        owner: str,
        repo: str,
        query_text: str,
        top_k: int = 10,
    ) -> list[RetrievedChunk]:
        """Search for chunks similar to the query.

        Args:
            owner: GitHub repo owner.
            repo: GitHub repo name.
            query_text: The user's question or search query.
            top_k: Maximum number of results.

        Returns:
            List of RetrievedChunk, sorted by relevance (best first).
        """
        collection = self.get_or_create_collection(owner, repo)

        if collection.count() == 0:
            logger.warning("Collection is empty — did you index first?")
            return []

        results = collection.query(
            query_texts=[query_text],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        # ChromaDB returns lists-of-lists (one per query text)
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            chunks.append(RetrievedChunk(
                chunk_id=chunk_id,
                content=doc,
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                language=meta.get("language", ""),
                distance=dist,
                symbol_name=meta.get("symbol_name", ""),
                symbol_type=meta.get("symbol_type", ""),
                parent_class=meta.get("parent_class", ""),
                imports=meta.get("imports", ""),
            ))

        logger.info("Query returned %d results", len(chunks))
        return chunks

    def delete_collection(self, owner: str, repo: str) -> None:
        """Delete a repo's collection (for re-indexing from scratch)."""
        name = self._collection_name(owner, repo)
        try:
            self.client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except ValueError:
            logger.warning("Collection '%s' not found", name)
