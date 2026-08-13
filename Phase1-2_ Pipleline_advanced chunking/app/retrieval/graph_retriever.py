import networkx as nx

class GraphRetriever:
    """
    A lightweight, in-memory Knowledge Graph Retriever using NetworkX.
    For this MVP, we build a static graph containing key entities from the project.
    In a production system, entities and edges would be extracted dynamically using an LLM.
    """
    def __init__(self):
        self.graph = nx.Graph()
        self._build_mock_graph()
        
    def _build_mock_graph(self):
        # Nodes and edges representing the 'RAG Fundamentals' domain
        relationships = [
            ("RAG", "LLM", "cung cấp ngữ cảnh cho"),
            ("RAG", "Vector Database", "sử dụng để lưu trữ nhúng"),
            ("Chunking", "Vector Database", "được lưu trữ thành từng đoạn nhỏ trong"),
            ("Embedding", "Vector Database", "tạo ra vector số để lưu vào"),
            ("DoRA", "LoRA", "là bản nâng cấp phân rã trọng số của"),
            ("DoRA", "Fine-Tuning", "đạt hiệu năng gần bằng với"),
            ("Reciprocal Rank Fusion", "Hybrid Search", "thuật toán cốt lõi dùng để trộn kết quả của"),
            ("BM25", "Hybrid Search", "là thuật toán tìm kiếm từ khóa trong"),
            ("Vector Search", "Hybrid Search", "là thuật toán tìm kiếm ngữ nghĩa trong"),
            ("Công ty A", "Công ty B", "bán 20% cổ phần cho"),
            ("Ông C", "Công ty B", "là tổng giám đốc của")
        ]
        
        for source, target, relation in relationships:
            self.graph.add_edge(source, target, relation=relation)
            
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Searches the graph by looking for entities mentioned in the query,
        then retrieves their 1-hop neighbors and relationships.
        """
        query_lower = query.lower()
        found_entities = [node for node in self.graph.nodes if node.lower() in query_lower]
        
        if not found_entities:
            return []
            
        results = []
        for entity in found_entities:
            neighbors = list(self.graph.neighbors(entity))
            for neighbor in neighbors:
                relation = self.graph.edges[entity, neighbor]['relation']
                text = f"[Graph Knowledge] '{entity}' {relation} '{neighbor}'."
                results.append({
                    "chunk_id": f"graph_{entity}_{neighbor}",
                    "text": text,
                    "source": "NetworkX Knowledge Graph",
                    "similarity": 1.0
                })
                
        return results[:top_k]
