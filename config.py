import os
from typing import Dict, Optional
from dotenv import load_dotenv


class Config:
    """
    配置管理类，用于加载和提供应用程序配置。
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        """
        从环境变量加载配置。
        """
        load_dotenv()
        self.base_url = os.environ.get('BASE_URL')
        self.api_key = os.environ.get('API_KEY')
        self.model = os.environ.get('MODEL')

    @classmethod
    def get_config(cls, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None) -> \
    Dict[str, str]:
        """
        获取标准格式的配置。

        Args:
            base_url (Optional[str]): 基础URL，如果提供则覆盖环境变量。
            api_key (Optional[str]): API密钥，如果提供则覆盖环境变量。
            model (Optional[str]): 模型名称，如果提供则覆盖环境变量。

        Returns:
            Dict[str, str]: 包含配置信息的字典。
        """
        config = cls()
        return {
            'base_url': base_url or config.base_url,
            'api_key': api_key or config.api_key,
            'model': model or config.model
        }

    @classmethod
    def get_openai_like_config(cls, base_url: Optional[str] = None, api_key: Optional[str] = None,
                               model: Optional[str] = None) -> Dict[str, str]:
        """
        获取OpenAI兼容格式的配置。

        Args:
            base_url (Optional[str]): 基础URL，如果提供则覆盖环境变量。
            api_key (Optional[str]): API密钥，如果提供则覆盖环境变量。
            model (Optional[str]): 模型名称，如果提供则覆盖环境变量。

        Returns:
            Dict[str, str]: 包含OpenAI兼容格式配置信息的字典。
        """
        config = cls()
        return {
            'api_base': base_url or config.base_url,
            'api_key': api_key or config.api_key,
            'model': model or config.model
        }
