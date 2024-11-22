__all__ = [
    "copy_param_spec",
    "copy_member_spec",
    "asyncify",
    "asyncify_thread",
    "asyncify_type",
    "syncify_type",
    "safe_format",
    "path_to_base64"
]

import re
import base64
import asyncio
import mimetypes
from pathlib import Path

from typing import (
    Optional, 
    List, 
    Dict,
    Any,
    Awaitable, 
    Callable, 
    Concatenate
)

def copy_param_spec[**P, R](f: Callable[P, R]) -> Callable[[Callable[..., Any]], Callable[P, R]]:
    return lambda x: x

def copy_member_spec[T, U, **P, R](f: Callable[Concatenate[T, P], R]) -> Callable[[Callable[..., Any]], Callable[Concatenate[U, P], R]]:
    return lambda x: x

def asyncify[**P, R](f: Callable[P, R]):
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return f(*args, **kwargs)
    return wrapper

def asyncify_thread[**P, R](f: Callable[P, R]):
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await asyncio.to_thread(f, *args, **kwargs)
    return wrapper

def asyncify_type[**P, R](f: Callable[P, R]) -> Callable[P, Awaitable[R]]:
    return f # type: ignore

def syncify_type[**P, R](f: Callable[P, Awaitable[R]]) -> Callable[P, R]:
    return f # type: ignore

def safe_format(text: str, 
                replacements: Dict[str, Any], 
                pattern: str = r'\{([a-zA-Z0-9_]+)\}', 
                strict: bool = False) -> str:
    matches = set(re.findall(pattern, text))
    if strict and (missing := matches - set(replacements.keys())):
        raise ValueError(f"Missing replacements for: {', '.join(missing)}")

    for match in matches & set(replacements.keys()):
        text = re.sub(r'\{' + match + r'\}', str(replacements[match]), text)
    return text

def path_to_base64(file_path: Path | str) -> str:
    mime_type, _ = mimetypes.guess_type(file_path)
    
    with open(file_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"

def to_lmc(
    message: str,
    attachments: List[str] | str | None = [],
    attachments_type: List[str] | str | None = "image_url",
    role: str = "user",
    author: Optional[str] = None,
    type: Optional[str] = "text"
) -> Dict[str, str | List[str | Any] | None]:
        
    if attachments:
        if isinstance(attachments, str):
            attachments = [attachments]
        if isinstance(attachments_type, str):
            attachments_type = [attachments_type] * len(attachments)
        
        if len(attachments) != len(attachments_type or []):
            raise ValueError("`attachments` and `attachments_type` must have the same length")

        attachments = [{"type": type, type: message}] + [
            {"type": att_type, att_type: {"url": att} }
            for att, att_type in zip(attachments, attachments_type)
        ]

    return {
        "role": role,
        "content": attachments or message,
        "author": author,
        "type": type
    }
    
def to_send(message: Dict[str, str | List[str] | None] | str, 
            *args: Any, **kwargs: Any) -> Dict[str, str | List[Dict[str, str]]]:
    
    if isinstance(message, str): 
        message = to_lmc(message, *args, **kwargs)    
    author = f"(name: {message['author']}) " if message['author'] else ""
    
    if isinstance(message["content"], list):
        message["content"][0]["text"] = author + message['content'][0]['text']
    else:
        message["content"] = author + message['content']
    return message
