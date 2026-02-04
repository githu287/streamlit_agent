from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
try:
    from langchain_community.chat_models.tongyi import BaseChatModel
except ImportError:
    from langchain_core.language_models import BaseChatModel as BaseChatModel
from utils.config_handler import rag_conf
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    """聊天模型工厂"""
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成硅基流动聊天模型实例"""
        return ChatOpenAI(model=rag_conf["chat_model_name"], base_url=rag_conf["siliconflow_base_url"],
                api_key=rag_conf["siliconflow_api_key"])


class EmbeddingsFactory(BaseModelFactory):
    """嵌入模型工厂"""
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        """生成硅基流动嵌入模型实例"""
        return OpenAIEmbeddings(model=rag_conf["embedding_model_name"], base_url=rag_conf["siliconflow_base_url"],
                api_key=rag_conf["siliconflow_api_key"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
