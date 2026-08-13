"""
Statutory Citation Graph Parser & Linker for Legal RAG:
Detects explicit legal cross-references (e.g., 'Điều 15', 'Khoản 2', 'Section 4.1', 'Clause 12')
and links candidate chunks to their cited reference chunks in a directed 1-hop citation graph.
"""
import re
from typing import List, Dict, Set


CITATION_PATTERNS = [
    r"(Điều\s+\d+)",
    r"(Khoản\s+\d+)",
    r"(Mục\s+\d+)",
    r"(Section\s+\d+(?:\.\d+)?)",
    r"(Clause\s+\d+(?:\.\d+)?)",
    r"(Article\s+\d+)",
]


class LegalCitationGraph:
    def __init__(self):
        self.chunk_citations: Dict[str, Set[str]] = {}
        self.reference_to_chunks: Dict[str, List[str]] = {}

    def index_chunk(self, chunk_id: str, text: str):
        citations = set()
        for pat in CITATION_PATTERNS:
            matches = re.findall(pat, text, flags=re.IGNORECASE)
            for m in matches:
                citations.add(m.strip())

        self.chunk_citations[chunk_id] = citations

        for cite in citations:
            cite_key = cite.lower()
            if cite_key not in self.reference_to_chunks:
                self.reference_to_chunks[cite_key] = []
            if chunk_id not in self.reference_to_chunks[cite_key]:
                self.reference_to_chunks[cite_key].append(chunk_id)

    def get_related_chunks(self, candidate_chunk_ids: List[str]) -> List[str]:
        """
        1-hop graph expansion: finds all chunks that share legal citation references
        with the given candidate chunk IDs.
        """
        expanded = set(candidate_chunk_ids)
        for cid in candidate_chunk_ids:
            citations = self.chunk_citations.get(cid, set())
            for cite in citations:
                cite_key = cite.lower()
                related_ids = self.reference_to_chunks.get(cite_key, [])
                for rid in related_ids:
                    expanded.add(rid)

        return list(expanded)
