from dataclasses import dataclass


@dataclass
class RouterResult:

    strategy: str
    reason: str
    confidence: float | None = None
    router_type: str = "rule_based"