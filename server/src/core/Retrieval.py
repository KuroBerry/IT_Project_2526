import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import text_to_sparse_vector_bm25

class Retrieval:
    def __init__(self, pc, dense_index, sparse_index, embedding_model = None, bm25 = None, vocabulary = None):
        self.pc = pc
        self.dense_index = dense_index
        self.sparse_index = sparse_index
        self.embedding_model = embedding_model
        self.bm25 = bm25
        self.vocabulary = vocabulary

    def semantic_search(self, query, namespace, top_k_dense):
        dense_results = self.dense_index.query(
            namespace= namespace,
            vector=self.embedding_model.encode(query, convert_to_tensor=False).tolist(),
            top_k=top_k_dense,
            include_metadata=True,
        )
        return dense_results
        # print(dense_results)
        
    def lexical_search(self, query, namespace, top_k_sparse):
        sparse_vector = text_to_sparse_vector_bm25(query, self.bm25, self.vocabulary)
        sparse_results = self.sparse_index.query(
            namespace=namespace,
            sparse_vector=sparse_vector,
            top_k=top_k_sparse,
            include_metadata=True,
        )

        return sparse_results
        # print(sparse_results)

    def hybrid_search(self, query, namespace, top_k = 5):
        dense_results = self.semantic_search(query, namespace, top_k)
        sparse_results = self.lexical_search(query, namespace, top_k)

        deduped_hits = {hit['id']: hit for hit in dense_results['matches'] + sparse_results['matches']}.values()
        # Sort by _score descending
        sorted_hits = sorted(deduped_hits, key=lambda x: x['score'], reverse=True)
        # Transform to format for reranking
        combined_results = [{'id': hit['id'], 'content': hit['metadata']['content']} for hit in sorted_hits]


        return combined_results