from typing import Optional
import chromadb
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from Embedding import BGEEbedding
from llama_index.core.base.embeddings.base import BaseEmbedding


def save_index_to_chroma(path: str, collection: str):
    """
    返回一个上下文存储器

    Args:
        path (str): Chroma数据库的持久化路径。
        collection (str): 要使用的集合名称。

    """
    db = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_or_create_collection(collection)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return StorageContext.from_defaults(vector_store=vector_store)


def get_index_from_chroma(
    path: str, collection: str, embedding_model: Optional[BaseEmbedding] = None
) -> VectorStoreIndex:
    """
    从Chroma数据库创建并返回一个VectorStoreIndex。

    Args:
        path (str): Chroma数据库的持久化路径。
        collection (str): 要使用的集合名称。
        embedding_model (Optional[LangchainEmbedding]): 可选的嵌入模型。如果未提供，将创建一个新的EmbeddingModel。

    Returns:
        VectorStoreIndex: 创建的向量存储索引。

    Raises:
        chromadb.exceptions.CollectionNotFoundException: 如果指定的集合不存在。
    """
    # 如果没有提供嵌入模型，则创建一个新的
    if embedding_model is None:
        embedding_model = BGEEbedding()

    # 创建Chroma客户端并获取集合
    chroma_client = chromadb.PersistentClient(path)
    try:
        chroma_collection = chroma_client.get_collection(collection)
    except Exception as e:
        raise ValueError(f"Collection '{collection}' not found in the database.")

    # 创建ChromaVectorStore
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 创建并返回VectorStoreIndex
    return VectorStoreIndex.from_vector_store(
        vector_store=vector_store, embed_model=embedding_model
    )
