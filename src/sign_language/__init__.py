"""Research package for robust sign-language recognition."""

from .config.settings import Settings, get_settings


def __getattr__(name: str):
    if name in {"NoiseScenario", "evaluate_robustness"}:
        from .robustness import NoiseScenario, evaluate_robustness
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["NoiseScenario", "Settings", "evaluate_robustness", "get_settings"]

