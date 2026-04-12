"""Train / validation / test splits by hospital (site) for domain generalization."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def split_seen_cases_by_site(case_df, val_fraction, test_fraction, seed):
    """Split seen-domain cases per site into train / val / in-domain test sets."""
    train_case_ids, val_case_ids, test_case_ids = [], [], []
    for site, site_cases in case_df.groupby("site"):
        case_ids = site_cases["case_id"].drop_duplicates().to_numpy().copy()
        rng = np.random.default_rng(seed + int(site))
        rng.shuffle(case_ids)

        n_cases = len(case_ids)
        n_val = int(round(n_cases * val_fraction))
        n_test = int(round(n_cases * test_fraction))
        if n_cases >= 3:
            if n_val == 0:
                n_val = 1
            if n_test == 0:
                n_test = 1
            while n_val + n_test >= n_cases:
                if n_test >= n_val and n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
        elif n_cases == 2:
            n_val, n_test = 1, 0
        else:
            n_val, n_test = 0, 0

        val_case_ids.extend(case_ids[:n_val].tolist())
        test_case_ids.extend(case_ids[n_val:n_val + n_test].tolist())
        train_case_ids.extend(case_ids[n_val + n_test:].tolist())

    return set(train_case_ids), set(val_case_ids), set(test_case_ids)


def summarize_split(name: str, df: pd.DataFrame) -> None:
    """Print basic stats (cases, slices, tumor ratio) for a split."""
    case_count = df["case_id"].nunique()
    slice_count = int(df["num_slices"].sum())
    tumor_count = int(df["num_tumor"].sum())
    normal_count = int(df["num_normal"].sum())
    total_labeled = tumor_count + normal_count
    tumor_ratio = (tumor_count / total_labeled) if total_labeled else 0.0
    print(f"\n{name}")
    print(
        f"cases={case_count}, slices={slice_count}, tumor={tumor_count}, "
        f"normal={normal_count}, tumor_ratio={tumor_ratio:.4f}"
    )
    print("top sites by cases:")
    print(df.groupby("site")["case_id"].nunique().sort_values(ascending=False).head(10))


def make_loso_split(
    index_df: pd.DataFrame,
    held_out_site: int,
    val_fraction: float = config.VALIDATION_FRACTION,
    test_fraction: float = config.IN_DOMAIN_TEST_FRACTION,
    max_cases_per_site: int = config.MAX_CASES_PER_SITE,
    seed: int = config.EXP_SEED,
) -> dict[str, pd.DataFrame]:
    """Build leave-one-site-out split frames for a given held-out site."""
    seen_df = index_df[index_df["site"] != held_out_site].copy()
    held_out_df = index_df[index_df["site"] == held_out_site].copy()

    seen_cases = seen_df[["site", "case_id"]].drop_duplicates()
    seen_cases = pd.concat(
        [
            site_cases.sample(n=min(len(site_cases), max_cases_per_site), random_state=seed)
            for _, site_cases in seen_cases.groupby("site")
        ],
        ignore_index=True,
    )

    selected_case_ids = set(seen_cases["case_id"].tolist())
    selected_seen_df = seen_df[seen_df["case_id"].isin(selected_case_ids)].copy()
    train_case_ids, val_case_ids, in_domain_case_ids = split_seen_cases_by_site(
        seen_cases, val_fraction, test_fraction, seed
    )

    return {
        "train_df": selected_seen_df[selected_seen_df["case_id"].isin(train_case_ids)].copy(),
        "val_df": selected_seen_df[selected_seen_df["case_id"].isin(val_case_ids)].copy(),
        "in_domain_test_df": selected_seen_df[selected_seen_df["case_id"].isin(in_domain_case_ids)].copy(),
        "out_domain_test_df": held_out_df.copy(),
    }


def build_loso_site_case_table(
    index_df: pd.DataFrame, min_cases: int = config.MIN_CASES_FOR_LOSO
) -> pd.DataFrame:
    """Rank sites by case count and flag those large enough for LOSO."""
    case_table = (
        index_df[["site", "case_id"]]
        .drop_duplicates()
        .groupby("site")
        .size()
        .rename("cases")
        .reset_index()
    )
    case_table["candidate_for_loso"] = case_table["cases"] >= min_cases
    return case_table.sort_values(
        ["candidate_for_loso", "cases"], ascending=[False, False]
    ).reset_index(drop=True)


def build_site_domain_summary(
    index_df: pd.DataFrame, min_cases_for_loso: int = config.MIN_CASES_FOR_LOSO
) -> pd.DataFrame:
    """Full per-site summary (cases, slices, tumor ratio, LOSO-candidate flag)."""
    summary = (
        index_df.groupby("site")
        .agg(
            cases=("case_id", "nunique"),
            slices=("num_slices", "sum"),
            tumor=("num_tumor", "sum"),
            normal=("num_normal", "sum"),
        )
        .reset_index()
    )
    summary["tumor_ratio"] = summary["tumor"] / (summary["tumor"] + summary["normal"])
    summary["candidate_for_loso"] = summary["cases"] >= min_cases_for_loso
    return summary


def build_main_split(
    index_df: pd.DataFrame,
    test_sites: list[int],
    val_fraction: float = config.VALIDATION_FRACTION,
    test_fraction: float = config.IN_DOMAIN_TEST_FRACTION,
    max_cases_per_site: int = config.MAX_CASES_PER_SITE,
    seed: int = config.EXP_SEED,
):
    """Build the main held-out-site split used for MixStyle variant experiments.

    Returns
    -------
    train_df, val_df, in_domain_test_df, out_domain_test_df : DataFrames
    train_set, val_set, in_domain_test_set, out_domain_test_set : sets of case_ids
    """
    trainval = index_df[~index_df["site"].isin(test_sites)].copy()
    test = index_df[index_df["site"].isin(test_sites)].copy()

    seen_cases_df = trainval[["site", "case_id"]].drop_duplicates()
    seen_cases_limited = pd.concat(
        [
            site_cases.sample(n=min(len(site_cases), max_cases_per_site), random_state=seed)
            for _, site_cases in seen_cases_df.groupby("site")
        ],
        ignore_index=True,
    )

    selected_case_set = set(seen_cases_limited["case_id"].tolist())
    selected_seen_df = trainval[trainval["case_id"].isin(selected_case_set)].copy()

    train_set, val_set, in_domain_test_set = split_seen_cases_by_site(
        seen_cases_limited,
        val_fraction=val_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )

    train_df = selected_seen_df[selected_seen_df["case_id"].isin(train_set)].copy()
    val_df = selected_seen_df[selected_seen_df["case_id"].isin(val_set)].copy()
    in_domain_test_df = selected_seen_df[selected_seen_df["case_id"].isin(in_domain_test_set)].copy()
    out_domain_test_df = test.copy()
    out_domain_test_set = set(out_domain_test_df["case_id"].unique().tolist())

    return {
        "train_df": train_df,
        "val_df": val_df,
        "in_domain_test_df": in_domain_test_df,
        "out_domain_test_df": out_domain_test_df,
        "train_set": train_set,
        "val_set": val_set,
        "in_domain_test_set": in_domain_test_set,
        "out_domain_test_set": out_domain_test_set,
    }
