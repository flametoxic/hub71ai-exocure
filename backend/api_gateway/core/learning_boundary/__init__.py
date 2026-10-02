"""Мастер-ТЗ F1: граница обучения — фильтр невязок, очистка PII, типизированные кандидаты (прод не меняется)."""
from .boundary import (F_CALIBRATION, F_RESIDUAL, BoundaryPolicy, CandidateKind, FilterResult, FilterVerdict,
                       LearningBoundaryError, ResidualFilter, ResidualRecord, TypedCandidate, make_candidate, sanitize)

__all__ = ["BoundaryPolicy", "CandidateKind", "F_CALIBRATION", "F_RESIDUAL", "FilterResult", "FilterVerdict",
           "LearningBoundaryError", "ResidualFilter", "ResidualRecord", "TypedCandidate", "make_candidate", "sanitize"]
