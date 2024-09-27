from llama_index.llms.langchain import LangChainLLM
from langchain_openai import ChatOpenAI
from llama_index.embeddings.langchain import LangchainEmbedding
from Embedding import EmbeddingModel
from config import Config


def get_model():
    return LangChainLLM(ChatOpenAI(**Config.get_config()))

def get_embedding():
    return LangchainEmbedding(EmbeddingModel())
