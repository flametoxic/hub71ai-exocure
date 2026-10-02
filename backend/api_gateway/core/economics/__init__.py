"""ТЗ «EXO Economic Optimizer, Multi-Agent Reasoning & Cross-Domain Generalization».

  lifecycle      — J жизненного цикла, u* в безопасном множестве, NPV и риск-скорректированная NPV (§1.1–1.2);
  maintenance    — Q_a(u) для обслуживания/отсрочки/снижения нагрузки/замены/осмотра, рекомендация с ценой,
                   уверенностью, допущениями и последствиями отсрочки (§1.3);
  arbitration    — допустимое множество → Парето → веса политики и неравенство; предложения сторон (§2, E2);
  society        — подписанные утверждения агентов и агрегация не голосованием (§3);
  generalization — структурные аналогии, Genesis, формальные доказательства, цели, гейт возможностей (§4–§8);
  metrics        — экономия только из проверенных исходов (§10).
Ничего из этого не действует в мире: результаты — рекомендации, кандидаты и гипотезы для планирования и людей.
"""
from .arbitration import ArbitrationPolicy, Candidate, Proposal, Stakeholder, arbitrate
from .lifecycle import choose_plan, lifecycle_objective, npv, risk_adjusted_npv
from .maintenance import MaintenanceOption, recommend, weibull_failure_probability
from .society import AgentClaim, AgentKeyRegistry, aggregate

from .genesis import GenesisSandboxLoop, z3_entails

__all__ = ["GenesisSandboxLoop", "z3_entails", "AgentClaim", "AgentKeyRegistry", "ArbitrationPolicy", "Candidate", "MaintenanceOption", "Proposal",
           "Stakeholder", "aggregate", "arbitrate", "choose_plan", "lifecycle_objective", "npv", "recommend",
           "risk_adjusted_npv", "weibull_failure_probability"]
