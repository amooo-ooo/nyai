__all__ = [
    "Provider",
    "PROVIDERS"
]

from dataclasses import dataclass, field
from typing import Dict, Any
from .utils import safe_format

@dataclass
class Provider:
    name: str
    endpoint: str
    api_key: str | None = None
    _api_key: str | None = None
    version: int | float | str = 1
    function_calls: bool = False
    images: bool = False
    embeddings: bool = False
    schema: bool = False
    max_tokens: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    @property
    def api_key(self) -> str:
        return self._api_key or self.name.upper() + "_API_KEY"
    
    @api_key.setter
    def api_key(self, api_key: str) -> None:
        if isinstance(api_key, property):
            return
        self._api_key = api_key
        
    @property
    def endpoint(self) -> str:
        return safe_format(self._endpoint, {"version": f"v{self.version}"} | self.options)
    
    @endpoint.setter
    def endpoint(self, endpoint: str) -> None:
        if isinstance(endpoint, property):
            return
        self._endpoint = endpoint
    
PROVIDERS = {
    "openai": Provider(name="openai", endpoint="https://api.openai.com/{version}"),
    "sambanova": Provider(name="sambanova", endpoint="https://api.sambanova.ai/{version}"),
    "anthropic": Provider(name="anthropic", endpoint="https://api.anthropic.com/{version}"),
    "cohere": Provider(name="cohere", endpoint="https://api.cohere.com/{version}", version=2),
    "fireworks": Provider(name="fireworks", endpoint="https://api.fireworks.ai/inference/{version}"),
    "together": Provider(name="together", endpoint="https://api.together.xyz/{version}"),
    "vertex_ai": Provider(name="vertex_ai", endpoint="https://{region}-aiplatform.googleapis.com/{version}/projects/{project_id}/locations/{region}/endpoints/{endpoint}", version="1beta1")
}
