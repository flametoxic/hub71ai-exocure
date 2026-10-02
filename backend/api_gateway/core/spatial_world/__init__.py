"""Пространственная модель мира CURE (мастер-ТЗ A1–A4; Gap Closure v1 §3–§5, v2 §2–§5; WM Engineering Spec
§2.1–2.5, §2.9–2.10, §2.15–2.16; City WM schema §2.1–2.3).

Единый журнал мира → битемпоральные утверждения, реестр сущностей с разрешением имён, дерево систем
координат с историей, версионированная топология, геометрия L0–L3, видимость и покрытие датчиков,
слияние состояния с детектором дрейфа, динамические треки. Писать — только через ProjectionPipeline.
Отдельный пакет: модули команды в world_model (entity_resolution, projection_pipeline, service, …)
не затрагиваются.
"""
from .dynamic_state import DynamicStateService, TrackState
from .entities import (DETERMINISTIC_NAMESPACES, EntityIdentity, EntityRegistry, ResolutionResult, ResolutionStatus,
                       alias_similarity, normalize_alias)
from .frames import ROOT, FrameEdge, SpatialPose, SpatialReferenceService, translation
from .fusion import FusedEstimate, Reading, StateFusionService, categorical_fusion, inverse_variance_weighted
from .geometry import GeometryRef, GeometryService, SensorCoverageService, SensorView
from .journal import JournalRow, WorldJournal
from .ledger import BitemporalLedger, CanonicalWriteViolation, ConflictRecord
from .pipeline import ProjectionPipeline
from .policy import SpatialWorldError, SpatialWorldPolicy
from .ports import WorldModelFactPort, WorldModelIdentityPort, ports
from .service import SpatialWorldModel, WorldSnapshot
from .topology import EDGE_TYPES, PHYSICAL_TYPES, TopologyEdge, TopologyService

from .discovery import F_BOUNDED_PAIRS, admit_discovered_edge, enumerate_candidate_pairs
from .replay import DecisionReplayEngine, ReplayPack

__all__ = ["DecisionReplayEngine", "ReplayPack", "F_BOUNDED_PAIRS", "admit_discovered_edge", "enumerate_candidate_pairs", "BitemporalLedger", "CanonicalWriteViolation", "ConflictRecord", "DETERMINISTIC_NAMESPACES",
           "DynamicStateService", "EDGE_TYPES", "EntityIdentity", "EntityRegistry", "FrameEdge", "FusedEstimate",
           "GeometryRef", "GeometryService", "JournalRow", "PHYSICAL_TYPES", "ProjectionPipeline", "ROOT", "Reading",
           "ResolutionResult", "ResolutionStatus", "SensorCoverageService", "SensorView", "SpatialPose",
           "SpatialReferenceService", "SpatialWorldError", "SpatialWorldModel", "SpatialWorldPolicy",
           "StateFusionService", "TopologyEdge", "TopologyService", "TrackState", "WorldJournal", "WorldModelFactPort",
           "WorldModelIdentityPort", "WorldSnapshot", "ports",
           "alias_similarity", "categorical_fusion", "inverse_variance_weighted", "normalize_alias", "translation"]
