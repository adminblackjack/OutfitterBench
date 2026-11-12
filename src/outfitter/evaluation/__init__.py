"""Offline trajectory metrics. No actor interfaces or model calls."""

FORMAT = "outfitter-trajectory-metrics-v1"


def evaluate_run(*args, **kwargs):
    from .api import evaluate_run as evaluate
    return evaluate(*args, **kwargs)


def evaluate_run_cutoffs(*args, **kwargs):
    from .api import evaluate_run_cutoffs as evaluate
    return evaluate(*args, **kwargs)
