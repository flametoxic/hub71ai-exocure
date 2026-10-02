# STUB: минимальная JTMS
from dataclasses import dataclass, field
@dataclass
class Justification:
    node: str
    supporters: list
    kind: str
    value: float
@dataclass
class _Node:
    token: str
    value: float
    justification: Justification
    status: str = "in"
class TruthMaintenanceSystem:
    def __init__(self):
        self._nodes = {}
    def add_fact(self, token, value, justification):
        n = _Node(token, value, justification)
        n.status = "in" if all(self._nodes.get(s) is not None and self._nodes[s].status == "in" for s in justification.supporters) else "out"
        self._nodes[token] = n
    def get_node(self, token):
        return self._nodes.get(token)
    def retract_fact(self, token):
        affected, frontier = [], [token]
        while frontier:
            t = frontier.pop()
            n = self._nodes.get(t)
            if n is None or n.status == "out":
                continue
            n.status = "out"; affected.append(t)
            frontier += [k for k, m in self._nodes.items() if t in m.justification.supporters]
        return affected
