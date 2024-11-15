__all__ = [
    "safe_format",
    "path_to_base64"
]

import re
import base64
import mimetypes
from pathlib import Path

from typing import (
    Iterable, 
    Optional, 
    List, 
    Dict,
    Any
)

def safe_format(text, 
                replacements: Dict[str, Any], 
                pattern=r'\{([a-zA-Z0-9_]+)\}', 
                strict=False) -> str:
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
    attachments: Iterable[str] | str = [],
    attachments_type: Iterable[str] | str = "image_url",
    role: str = "user",
    author: Optional[str] = None,
    type: Optional[str] = "text"
) -> Dict[str, str | List[Dict[str, str]]]:
        
    if attachments:
        if isinstance(attachments, str):
            attachments = [attachments]
        if isinstance(attachments_type, str):
            attachments_type = [attachments_type] * len(attachments)
        
        if len(attachments) != len(attachments_type):
            raise ValueError("`attachments` and `attachments_type` must have the same length")

        attachments = [{"type": type, type: message}] + [
            {"type": att_type, att_type: {"url": att}}
            for att, att_type in zip(attachments, attachments_type)
        ]

    return {
        "role": role,
        "content": attachments or message,
        "author": author,
        "type": type
    }
    
def to_send(message: Dict[str, str | List[Dict[str, str]]] | str, 
            *args, **kwargs) -> Dict[str, str | List[Dict[str, str]]]:
    
    if isinstance(message, str): 
        message = to_lmc(message, *args, **kwargs)    
    author = f"(name: {message['author']}) " if message['author'] else ""
    
    if isinstance(message["content"], list):
        message["content"][0]["text"] = author + message['content'][0]['text']
    else:
        message["content"] = author + message['content']
    return message
