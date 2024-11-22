__all__ = [
    "Client",
    "AsyncClient"
]

from openai import OpenAI, AsyncOpenAI
from openai._types import NOT_GIVEN, Timeout
from openai._base_client import DEFAULT_MAX_RETRIES

from typing import Mapping, Dict
import httpx

from .providers import Provider, PROVIDERS
from .types import NotGiven
import os
            
class Client(OpenAI):
    def __init__(
        self, 
        provider: Provider | str = None,
        options: Dict = None,
        api_key: str | None = None, 
        organization: str | None = None, 
        project: str | None = None, 
        base_url: str | httpx.URL | None = None, 
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN, 
        max_retries: int = DEFAULT_MAX_RETRIES, 
        default_headers: Mapping[str, str] | None = None, 
        default_query: Mapping[str, object] | None = None, 
        http_client: httpx.Client | None = None, 
        _strict_response_validation: bool = False,
        ):
        
        if not isinstance(provider, Provider):
            provider = PROVIDERS.get(provider or "openai")
        if provider is None:
            raise ValueError("Provider not supported with nyai")
        self.provider = provider 
        self.provider.options |= options or {}
        
        if (api_key or os.getenv(self.provider.api_key)) is None:
            raise ValueError(f"Missing field `api_key` or env `{self.provider.api_key}`")
        
        super().__init__(
            api_key=api_key or os.getenv(self.provider.api_key), 
            organization=organization, 
            project=project, 
            base_url=base_url or self.provider.endpoint, 
            timeout=timeout, 
            max_retries=max_retries, 
            default_headers=default_headers, 
            default_query=default_query, 
            http_client=http_client, 
            _strict_response_validation=_strict_response_validation)

            
class AsyncClient(AsyncOpenAI):
    def __init__(
        self, 
        provider: Provider | str | None = None,
        options: Dict = None,
        api_key: str | None = None, 
        organization: str | None = None, 
        project: str | None = None, 
        base_url: str | httpx.URL | None = None, 
        timeout: float | Timeout | None | NotGiven = NOT_GIVEN, 
        max_retries: int = DEFAULT_MAX_RETRIES, 
        default_headers: Mapping[str, str] | None = None, 
        default_query: Mapping[str, object] | None = None, 
        http_client: httpx.Client | None = None, 
        _strict_response_validation: bool = False,
        ):
        
        if not isinstance(provider, Provider):
            provider = PROVIDERS.get(provider or "openai")
        if provider is None:
            raise ValueError("Provider not supported with nyai")
        self.provider = provider 
        self.provider.options |= options or {}
        
        if (api_key or os.getenv(self.provider.api_key)) is None:
            raise ValueError(f"Missing field `api_key` or env `{self.provider.api_key}`")
        
        super().__init__(
            api_key=api_key or os.getenv(self.provider.api_key), 
            organization=organization, 
            project=project, 
            base_url=base_url or self.provider.endpoint, 
            timeout=timeout, 
            max_retries=max_retries, 
            default_headers=default_headers, 
            default_query=default_query, 
            http_client=http_client, 
            _strict_response_validation=_strict_response_validation)