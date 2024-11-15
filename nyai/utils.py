__all__ = [
    "safe_format",
    "path_to_base64"
]

import re
from pathlib import Path
import base64
import mimetypes

def safe_format(text, replacements, pattern=r'\{([a-zA-Z0-9_]+)\}', strict=False):
    matches = set(re.findall(pattern, text))
    if strict and (missing := matches - set(replacements.keys())):
        raise ValueError(f"Missing replacements for: {', '.join(missing)}")

    for match in matches & set(replacements.keys()):
        text = re.sub(r'\{' + match + r'\}', str(replacements[match]), text)
    return text

def path_to_base64(file_path: Path | str):
    mime_type, _ = mimetypes.guess_type(file_path)
    
    with open(file_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode("utf-8")
    
    return f"data:{mime_type};base64,{base64_data}"
