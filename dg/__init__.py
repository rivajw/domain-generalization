"""Domain Generalization (BraTS + MixStyle) experimental library.

High-level entry points used by the notebook:

    from dg import config, runtime
    from dg.preprocessing import preprocess_zip_to_npz, visualize_dataset_samples
    from dg.splits import (
        build_main_split, build_site_domain_summary,
        build_loso_site_case_table, summarize_split,
    )
    from dg.experiment import (
        DataSplitConfig, ModelConfig, build_model,
        prepare_dataloaders_from_config,
        run_experiment, run_loso_experiment, run_loso_sweep,
        compare_models, report_experiments, load_trained_model,
    )
    from dg.training import set_seed, train_epoch, eval_epoch, eval_metrics
"""

from . import config, runtime
from .consistency import consistency_report, measure_consistency, set_all_mixstyle
from .dataset import BraTSNPZSliceDataset
from .experiment import (
    DataSplitConfig,
    ModelConfig,
    build_model,
    compare_models,
    load_trained_model,
    prepare_dataloaders_from_config,
    report_experiments,
    run_experiment,
    run_loso_experiment,
    run_loso_sweep,
)
from .models import MixStyle, ResNet18MixStyle
from .preprocessing import (
    download_from_synapse,
    preprocess_zip_to_npz,
    visualize_dataset_samples,
)
from .splits import (
    build_loso_site_case_table,
    build_main_split,
    build_site_domain_summary,
    make_loso_split,
    split_seen_cases_by_site,
    summarize_split,
)
from .training import eval_epoch, eval_metrics, set_seed, train_epoch, train_epoch_consistency

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
    # training
    "set_seed",
    "train_epoch",
    "train_epoch_consistency",
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
]
