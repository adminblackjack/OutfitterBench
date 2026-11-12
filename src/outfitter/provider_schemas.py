"""Lossless schema-envelope adaptations for compatible providers."""

from __future__ import annotations

import copy
from typing import Any


def poe_tool_schemas(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add redundant root properties required by Poe's schema validator."""

    result = copy.deepcopy(tools)
    for tool in result:
        schema = tool.get("function", {}).get("parameters", {})
        if schema.get("type") != "object" or "properties" in schema:
            continue
        branches = schema.get("oneOf", [])
        if not branches or not all(
            branch.get("type") == "object"
            and isinstance(branch.get("properties"), dict)
            and branch.get("additionalProperties") is False
            and not any(key in branch for key in ("patternProperties", "$ref"))
            for branch in branches
        ):
            raise ValueError(
                "cannot adapt a properties-free tool schema losslessly for Poe"
            )
        names = sorted(
            {name for branch in branches for name in branch["properties"]}
        )
        properties: dict[str, Any] = {}
        for name in names:
            alternatives = []
            for branch in branches:
                candidate = branch["properties"].get(name)
                if candidate is not None and candidate not in alternatives:
                    alternatives.append(copy.deepcopy(candidate))
            properties[name] = (
                alternatives[0]
                if len(alternatives) == 1
                else {"anyOf": alternatives}
            )
        schema["properties"] = properties
        common_required = set(branches[0].get("required", []))
        for branch in branches[1:]:
            common_required.intersection_update(branch.get("required", []))
        schema["required"] = sorted(
            set(schema.get("required", [])) | common_required
        )
    return result
