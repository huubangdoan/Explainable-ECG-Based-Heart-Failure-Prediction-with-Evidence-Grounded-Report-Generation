"""PTB-XL utilities; acquisition does not require the scientific dependencies."""

__all__ = ["CLASSES", "LEADS", "ECGDataset", "PipelineConfig", "audit_release",
           "build_manifest", "fit_normalizer", "iter_batches", "process_signals"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(name)
    from . import core
    return getattr(core, name)
