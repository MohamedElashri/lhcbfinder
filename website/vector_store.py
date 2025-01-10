# vector_store.py
from abc import ABC, abstractmethod
import os
from typing import List, Tuple, Dict, Any, Optional
import qdrant_client
from qdrant_client.http import models
from qdrant_client.http.models import Filter
from pinecone import Pinecone

class VectorStore(ABC):
    @abstractmethod
    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Upsert vectors into the store"""
        pass
    
    @abstractmethod
    def get_total_vectors(self) -> int:
        """Get total number of vectors in the store"""
        pass
    
    @abstractmethod
    def query(self, vector: Optional[List[float]] = None, id: Optional[str] = None, 
             top_k: int = 10, include_metadata: bool = True) -> Dict:
        """Query the vector store"""
        pass
    
    @abstractmethod
    def fetch(self, ids: List[str]) -> Dict:
        """Fetch vectors by IDs"""
        pass

class PineconeStore(VectorStore):
    def __init__(self, index_name: str):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.index = self.pc.Index(index_name)
    
    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        self.index.upsert(vectors=vectors)
    
    def get_total_vectors(self) -> int:
        stats = self.index.describe_index_stats()
        return stats["total_vector_count"]
        
    def query(self, vector: Optional[List[float]] = None, id: Optional[str] = None, 
             top_k: int = 10, include_metadata: bool = True) -> Dict:
        if vector is not None:
            return self.index.query(vector=vector, top_k=top_k, include_metadata=include_metadata)
        elif id is not None:
            return self.index.query(id=id, top_k=top_k, include_metadata=include_metadata)
        raise ValueError("Either vector or id must be provided")
    
    def fetch(self, ids: List[str]) -> Dict:
        return self.index.fetch(ids)

class QdrantStore(VectorStore):
    def __init__(self, collection_name: str, vector_size: int = 1024):
        """Initialize Qdrant client with local storage"""
        self.client = qdrant_client.QdrantClient(
            path="./qdrant_data"  # Local storage path
        )
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )
    
    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        points = [
            models.PointStruct(
                id=id_hash(id_),
                vector=vector,
                payload=metadata
            )
            for id_, vector, metadata in vectors
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def get_total_vectors(self) -> int:
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.points_count
        
    def query(self, vector: Optional[List[float]] = None, id: Optional[str] = None, 
             top_k: int = 10, include_metadata: bool = True) -> Dict:
        if vector is not None:
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                limit=top_k
            )
            # Convert to Pinecone-like format
            matches = [{
                "id": str(r.id),
                "score": float(r.score),
                "metadata": r.payload
            } for r in results]
            return {"matches": matches}
            
        elif id is not None:
            results = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[id_hash(id)]
            )
            if not results:
                return {"matches": []}
            # Convert to Pinecone-like format
            matches = [{
                "id": str(r.id),
                "score": 1.0,  # For direct ID lookup
                "metadata": r.payload
            } for r in results]
            return {"matches": matches}
            
        raise ValueError("Either vector or id must be provided")
    
    def fetch(self, ids: List[str]) -> Dict:
        numeric_ids = [id_hash(id_) for id_ in ids]
        results = self.client.retrieve(
            collection_name=self.collection_name,
            ids=numeric_ids
        )
        return {
            "vectors": {
                str(r.id): {
                    "id": str(r.id),
                    "metadata": r.payload
                } for r in results
            }
        }

class MultiVectorStore(VectorStore):
    """Store vectors in multiple backends simultaneously"""
    def __init__(self, stores: List[VectorStore]):
        self.stores = stores
    
    def upsert(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        for store in self.stores:
            store.upsert(vectors)
    
    def get_total_vectors(self) -> int:
        # Return count from first store
        return self.stores[0].get_total_vectors() if self.stores else 0
        
    def query(self, vector: Optional[List[float]] = None, id: Optional[str] = None, 
             top_k: int = 10, include_metadata: bool = True) -> Dict:
        # Use first store for queries
        return self.stores[0].query(vector=vector, id=id, top_k=top_k, 
                                  include_metadata=include_metadata)
    
    def fetch(self, ids: List[str]) -> Dict:
        # Use first store for fetches
        return self.stores[0].fetch(ids)

def id_hash(id_str: str) -> int:
    """Convert string ID to uint64 for Qdrant"""
    return hash(id_str) & 0xFFFFFFFFFFFFFFFF

def create_vector_store(use_pinecone: bool = True, 
                       use_qdrant: bool = False,
                       pinecone_index: Optional[str] = None,
                       qdrant_collection: Optional[str] = None,
                       vector_size: int = 1024) -> VectorStore:
    """Factory function to create vector store instance(s)"""
    stores = []
    
    if use_pinecone and pinecone_index:
        stores.append(PineconeStore(pinecone_index))
    
    if use_qdrant and qdrant_collection:
        stores.append(QdrantStore(qdrant_collection, vector_size))
    
    if not stores:
        raise ValueError("At least one vector store must be enabled")
        
    return MultiVectorStore(stores) if len(stores) > 1 else stores[0]