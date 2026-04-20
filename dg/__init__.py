"""Domain Generalization (BraTS + MixStyle) experimental library.

This package intentionally keeps top-level imports light so notebooks can do
``from dg import config, runtime`` without eagerly importing the full training
stack (and, transitively, ``torchvision``). Heavier symbols are loaded on first
attribute access via :func:`__getattr__`.
"""

from importlib import import_module

from . import config, runtime

_LAZY_EXPORTS = {
    # dataset / model
    "BraTSNPZSliceDataset": (".dataset", "BraTSNPZSliceDataset"),
    "MixStyle": (".models", "MixStyle"),
    "ResNet18MixStyle": (".models", "ResNet18MixStyle"),
    # preprocessing
    "download_from_synapse": (".preprocessing", "download_from_synapse"),
    "preprocess_zip_to_npz": (".preprocessing", "preprocess_zip_to_npz"),
    "visualize_dataset_samples": (".preprocessing", "visualize_dataset_samples"),
    # splits
    "split_seen_cases_by_site": (".splits", "split_seen_cases_by_site"),
    "make_loso_split": (".splits", "make_loso_split"),
    "build_loso_site_case_table": (".splits", "build_loso_site_case_table"),
    "build_site_domain_summary": (".splits", "build_site_domain_summary"),
    "build_main_split": (".splits", "build_main_split"),
    "summarize_split": (".splits", "summarize_split"),
    # consistency
    "set_all_mixstyle": (".consistency", "set_all_mixstyle"),
    "measure_consistency": (".consistency", "measure_consistency"),
    "consistency_report": (".consistency", "consistency_report"),
    "multiscale_feature_consistency": (".consistency", "multiscale_feature_consistency"),
    # training
    "set_seed": (".training", "set_seed"),
    "train_epoch": (".training", "train_epoch"),
    "train_epoch_consistency": (".training", "train_epoch_consistency"),
    "train_epoch_consistency_multiscale": (".training", "train_epoch_consistency_multiscale"),
    "eval_epoch": (".training", "eval_epoch"),
    "eval_metrics": (".training", "eval_metrics"),
    # experiment
    "DataSplitConfig": (".experiment", "DataSplitConfig"),
    "ModelConfig": (".experiment", "ModelConfig"),
    "build_model": (".experiment", "build_model"),
    "prepare_dataloaders_from_config": (".experiment", "prepare_dataloaders_from_config"),
    "run_experiment": (".experiment", "run_experiment"),
    "run_loso_experiment": (".experiment", "run_loso_experiment"),
    "run_loso_sweep": (".experiment", "run_loso_sweep"),
    "compare_models": (".experiment", "compare_models"),
    "report_experiments": (".experiment", "report_experiments"),
    "load_trained_model": (".experiment", "load_trained_model"),
    "load_experiment_result": (".experiment", "load_experiment_result"),
}


def __getattr__(name: str):
    if name in _LAZY_EXPORTS:
        module_name, attr_name = _LAZY_EXPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # modules
    "config",
    "runtime",
    # dataset / model
    "BraTSNPZSliceDataset",
    "MixStyle",
    "ResNet18MixStyle",
    # preprocessing
    "download_from_synapse",
    "preprocess_zip_to_npz",
    "visualize_dataset_samples",
    # splits
    "split_seen_cases_by_site",
    "make_loso_split",
    "build_loso_site_case_table",
    "build_site_domain_summary",
    "build_main_split",
    "summarize_split",
    # consistency
    "set_all_mixstyle",
    "measure_consistency",
    "consistency_report",
    "multiscale_feature_consistency",
    # training
    "set_seed",
    "train_epoch",
    "train_epoch_consistency",
    "train_epoch_consistency_multiscale",
    "eval_epoch",
    "eval_metrics",
    # experiment
    "DataSplitConfig",
    "ModelConfig",
    "build_model",
    "prepare_dataloaders_from_config",
    "run_experiment",
    "run_loso_experiment",
    "run_loso_sweep",
    "compare_models",
    "report_experiments",
    "load_trained_model",
    "load_experiment_result",
]
