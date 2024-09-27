import httpx
import requests
from langchain_core.embeddings import Embeddings
from typing import List, Dict, Any, Optional
import asyncio
from llama_index.core.base.embeddings.base import (
    BaseEmbedding,
    DEFAULT_EMBED_BATCH_SIZE,
    Embedding,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.bridge.pydantic import Field, PrivateAttr

TIMEOUT = 80
URL = "http://192.168.50.145:9016/embedding/bge"
CHUNK_SIZE = 80


class BGEEbedding(BaseEmbedding):
    url: str = Field(default=URL, description="""bge访问的接口地址""")
    chunk_size: int = Field(default=CHUNK_SIZE, description="""嵌入的chunksize""")

    def __init__(
            self,
            embed_betch_size: int = DEFAULT_EMBED_BATCH_SIZE,
            callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            embed_betch_size=embed_betch_size,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        return """BGEEbedding"""

    def _async_not_implemented_warn_once(self) -> None:
        if not self._async_not_implemented_warned:
            print("Async embedding not available, falling back to sync method.")
            self._async_not_implemented_warned = True

    async def _arequest(self, texts: List[str]) -> List[List[float]]:
        """发送异步请求获取嵌入"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url, json={"texts": texts}, timeout=TIMEOUT
            )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Async embedding request failed: {response.text}")

    def _process_batch(
            self, texts: List[str], is_async: bool = False
    ) -> List[List[float]]:
        """处理文本批次，获取嵌入"""
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i: i + self.chunk_size]
            if is_async:
                response = asyncio.run(self._arequest(batch))
            else:
                response = self._request(batch)
            embeddings.extend(response)
        return embeddings

    def _get_query_embedding(self, query: str) -> List[float]:
        """Get query embedding."""
        return self._process_batch([query])[0]

    async def _aget_query_embedding(self, query: str) -> Embedding:
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._process_batch, [query], True
        )
        return result[0]

    def _get_text_embedding(self, text: str) -> List[float]:
        """Get text embedding."""
        return self._process_batch([text])[0]

    async def _aget_text_embedding(self, text: str) -> Embedding:
        result = await asyncio.get_event_loop().run_in_executor(
            None, self._process_batch, [text], True
        )
        return result[0]

    def _get_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return self._process_batch(texts)

    async def _aget_text_embeddings(self, texts: List[str]) -> List[Embedding]:
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_batch, texts, True
        )

    def _request(self, texts: List[str]) -> List[List[float]]:
        """发送同步请求获取嵌入"""
        with requests.Session() as session:
            response = session.post(self.url, json={"texts": texts}, timeout=TIMEOUT)
        if response.ok:
            return response.json()
        raise Exception(f"Embedding request failed: {response.text}")


class LangchainBGEEbedding(Embeddings):
    """嵌入模型类，提供文本嵌入功能"""

    def __init__(
            self,
            url: str = "http://192.168.50.145:9016/embedding/bge",
            chunk_size: int = 60,
            timeout: int = 80,
    ):
        self.url = url
        self.chunk_size = chunk_size
        self.TIMEOUT = timeout

    def _request(self, texts: List[str]) -> List[List[float]]:
        """发送同步请求获取嵌入"""
        with requests.Session() as session:
            response = session.post(
                self.url, json={"texts": texts}, timeout=self.TIMEOUT
            )
        if response.ok:
            return response.json()
        raise Exception(f"Embedding request failed: {response.text}")

    async def _arequest(self, texts: List[str]) -> List[List[float]]:
        """发送异步请求获取嵌入"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.url, json={"texts": texts}, timeout=self.TIMEOUT
            )
        if response.status_code == 200:
            return response.json()
        raise Exception(f"Async embedding request failed: {response.text}")

    def _process_batch(
            self, texts: List[str], is_async: bool = False
    ) -> List[List[float]]:
        """处理文本批次，获取嵌入"""
        embeddings: List[List[float]] = []
        for i in range(0, len(texts), self.chunk_size):
            batch = texts[i: i + self.chunk_size]
            if is_async:
                response = asyncio.run(self._arequest(batch))
            else:
                response = self._request(batch)
            embeddings.extend(response)
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档"""
        return self._process_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        """嵌入单个查询"""
        return self.embed_documents([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """异步嵌入多个文档"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._process_batch, texts, True
        )

    async def aembed_query(self, text: str) -> List[float]:
        """异步嵌入单个查询"""
        result = await self.aembed_documents([text])
        return result[0]


if __name__ == "__main__":
    print(BGEEbedding().get_text_embedding("""你好"""))
