"""Feature normalization and building modules."""

from .normalization import normalize_landmarks
from .representations import (
	FeatureSpecification,
	get_feature_specification,
	resolve_feature_representation,
	select_feature_representation,
)

__all__ = [
	"FeatureSpecification",
	"get_feature_specification",
	"normalize_landmarks",
	"resolve_feature_representation",
	"select_feature_representation",
]
