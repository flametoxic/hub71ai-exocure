"""WM-05 Operational Integrity, Site Compilation & Deployable Reality.

§1 canonical writes + ProjectionContext + эфемерные контексты · §2 карантин (spatial_world.entities) ·
§3–4 владельцы представлений, контракт невязки, лестница прогнозов, допуск нейромодели · §5 причинная целостность ·
§6 Reality Compiler (spatial_world.site_compiler) · §7 data_mode · §8 состязательный корпус (benchmark.adversarial) ·
§9 совместимость версий · §10 порядок событий · §11 операторы · §12 надёжность · §13 цепочка поставки.
"""
from .causal_integrity import (CausalCoverageReport, CausalMethod, CausalResultEnvelope, DecisionPath,
                               IdentificationStatus, ObservationalAsInterventionalError, causal_coverage, coverage_gate,
                               merge_results, trust_weighted_sigma, validate_elimination)
from .compatibility import Compatibility, CompatibilityMatrix, VersionVector
from .context import (CanonicalWriteViolation, DataMode, EphemeralSessionContext, IntegrityError, ProjectionContext,
                      require_data_mode, truthful_label, weakest_mode)
from .ordering import EventOrderingService, OrderedEvent
from .organization import (Artifact, Operator, Organization, ReliabilityRegistry, ShiftWindow, SupplyChain,
                           availability, feasible_actions)
from .prediction import (NeuralPromotionEvidence, PredictionEnvelope, PredictionResidual, PredictionSource,
                         promote_neural_model, require_owner, residual_to_signal, select_prediction_source)

from .degradation import RESTRICTIONS, DegradationManager, DegradationStatus

__all__ = ["DegradationManager", "DegradationStatus", "RESTRICTIONS", "Artifact", "CanonicalWriteViolation", "CausalCoverageReport", "CausalMethod", "CausalResultEnvelope",
           "Compatibility", "CompatibilityMatrix", "DataMode", "DecisionPath", "EphemeralSessionContext",
           "EventOrderingService", "IdentificationStatus", "IntegrityError", "NeuralPromotionEvidence",
           "ObservationalAsInterventionalError", "Operator", "OrderedEvent", "Organization", "PredictionEnvelope",
           "PredictionResidual", "PredictionSource", "ProjectionContext", "ReliabilityRegistry", "ShiftWindow",
           "SupplyChain", "VersionVector", "availability", "causal_coverage", "coverage_gate", "feasible_actions",
           "merge_results", "promote_neural_model", "require_data_mode", "require_owner", "residual_to_signal",
           "select_prediction_source", "trust_weighted_sigma", "truthful_label", "validate_elimination",
           "weakest_mode"]
