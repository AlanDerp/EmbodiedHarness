"""ASPIRE-style public tool registry shared by RPC and generated-program primitives."""

from dataclasses import dataclass
import inspect
import re


@dataclass(frozen=True)
class Tool:
    name: str
    function: object
    effect: str
    description: str


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, function, *, effect="read", description=""):
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name in self._tools:
            raise ValueError("invalid or duplicate tool name")
        if effect not in ("read", "action", "control") or not callable(function):
            raise ValueError("tool needs callable and explicit effect")
        self._tools[name] = Tool(name, function, effect, description)
        return self

    def methods(self):
        return {name: tool.function for name, tool in self._tools.items()}

    def describe(self):
        return [{"name": tool.name, "effect": tool.effect,
                 "signature": str(inspect.signature(tool.function)), "description": tool.description}
                for tool in self._tools.values()]

    def invoke(self, name, params):
        tool = self._tools[name]
        inspect.signature(tool.function).bind(**params)
        return tool.function(**params)
