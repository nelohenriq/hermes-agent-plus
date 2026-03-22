"""
TOON (Token-Oriented Object Notation) - Compact data serialization format.

TOON is a compact, readable format optimized for LLM token efficiency:
- Compact representation of nested data structures
- Preserves readability for humans while saving tokens
- Supports nested dictionaries, lists, and primitive types
- Used by prompt caching and statistics reporting

Format example:
```
name: John
age: 30
address: {
  street: 123 Main St
  city: Anytown
  state: CA
}
hobbies: [
  - reading
  - coding
  - hiking
]
```

Usage:
    from agent.toon import to_toon, from_toon

    # Serialize to TOON
    toon_str = to_toon({"name": "John", "age": 30})

    # Parse from TOON
    data = from_toon(toon_str)
"""

import re
from typing import Any, Dict, List, Union


def to_toon(data: Any, indent: int = 0) -> str:
    """
    Convert Python data structures to TOON format.

    Args:
        data: Python object (dict, list, primitive)
        indent: Current indentation level

    Returns:
        TOON formatted string
    """
    indent_str = "  " * indent

    if isinstance(data, dict):
        if not data:
            return "{}"

        lines = ["{"]
        for key, value in data.items():
            lines.append(f"{indent_str}  {key}: {to_toon(value, indent + 1)}")
        lines.append(f"{indent_str}}}")

        return "\n".join(lines)

    elif isinstance(data, list):
        if not data:
            return "[]"

        lines = ["["]
        for item in data:
            lines.append(f"{indent_str}  - {to_toon(item, indent + 1)}")
        lines.append(f"{indent_str}]")

        return "\n".join(lines)

    else:
        # Primitive types
        if isinstance(data, str):
            # Escape newlines and special chars in strings
            escaped = data.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")
            return f'"{escaped}"' if (" " in escaped or "\n" in escaped or "\t" in escaped) else escaped
        elif isinstance(data, bool):
            return "true" if data else "false"
        elif data is None:
            return "null"
        else:
            return str(data)


def from_toon(toon_str: str) -> Any:
    """
    Parse TOON format back to Python data structures.

    Args:
        toon_str: TOON formatted string

    Returns:
        Python object (dict, list, primitive)
    """
    # Remove comments and normalize whitespace
    lines = []
    for line in toon_str.split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        lines.append(line)

    return _parse_toon_lines(lines)


def _parse_toon_lines(lines: List[str], start_idx: int = 0) -> Any:
    """Parse TOON lines recursively."""
    if start_idx >= len(lines):
        return None

    line = lines[start_idx]

    # Dictionary
    if line.endswith('{'):
        result = {}
        idx = start_idx + 1
        while idx < len(lines):
            if lines[idx].rstrip() == '}':
                break

            # Parse key: value
            if ':' in lines[idx]:
                key_part, value_part = lines[idx].split(':', 1)
                key = key_part.strip()
                value_part = value_part.strip()

                if value_part == '{':
                    # Nested dict
                    value, idx = _parse_nested_dict(lines, idx)
                elif value_part == '[':
                    # Nested list
                    value, idx = _parse_nested_list(lines, idx)
                else:
                    # Simple value
                    value = _parse_primitive(value_part)
                    idx += 1

                result[key] = value
            else:
                idx += 1

        return result, idx + 1

    # List
    elif line.endswith('['):
        result = []
        idx = start_idx + 1
        while idx < len(lines):
            if lines[idx].rstrip() == ']':
                break

            # Parse - item
            if lines[idx].startswith('- '):
                item_part = lines[idx][2:].strip()

                if item_part == '{':
                    # Nested dict in list
                    value, idx = _parse_nested_dict(lines, idx)
                    result.append(value)
                elif item_part == '[':
                    # Nested list in list
                    value, idx = _parse_nested_list(lines, idx)
                    result.append(value)
                else:
                    # Simple value
                    result.append(_parse_primitive(item_part))
                    idx += 1
            else:
                idx += 1

        return result, idx + 1

    # Simple value
    else:
        return _parse_primitive(line), start_idx + 1


def _parse_nested_dict(lines: List[str], start_idx: int) -> tuple:
    """Parse a nested dictionary starting at the given index."""
    return _parse_toon_lines(lines, start_idx)


def _parse_nested_list(lines: List[str], start_idx: int) -> tuple:
    """Parse a nested list starting at the given index."""
    return _parse_toon_lines(lines, start_idx)


def _parse_primitive(value_str: str) -> Any:
    """Parse primitive values (strings, numbers, booleans, null)."""
    value_str = value_str.strip()

    # Handle quoted strings
    if value_str.startswith('"') and value_str.endswith('"'):
        escaped = value_str[1:-1]
        return escaped.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")

    # Handle unquoted strings and other types
    if value_str == "true":
        return True
    elif value_str == "false":
        return False
    elif value_str == "null":
        return None
    elif value_str.isdigit():
        return int(value_str)
    elif re.match(r'^-?\d+\.\d+$', value_str):
        return float(value_str)
    else:
        # String - unescape if needed
        return value_str.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def estimate_toon_tokens(toon_str: str) -> int:
    """
    Estimate token count for TOON string.

    Rough estimate: ~4 characters per token.

    Args:
        toon_str: TOON formatted string

    Returns:
        Estimated token count
    """
    return max(1, len(toon_str) // 4)