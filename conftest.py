import pytest
from _pytest.nodeid import NodeId

if not hasattr(NodeId(path=""), "_str_cache"):
    def _safe_str(self: NodeId) -> str:
        cache = getattr(self, "_str_cache", None)
        if cache is not None:
            return cache
        s = "::".join((self.path, *self.names))
        if self.params is not None:
            s += f"[{self.params}]"
        object.__setattr__(self, "_str_cache", s)
        return s

    NodeId.__str__ = _safe_str  # type: ignore[method-assign]
