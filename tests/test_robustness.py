from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ROBUSTNESS_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "robustness_summary.csv"
)

OVERLAP_SUMMARY_PATH = (
    PROJECT_ROOT
    / "reports"
    / "tables"
    / "overlap_diagnostics.csv"
)


def load_robustness_summary() -> pd.DataFrame:
    assert ROBUSTNESS_SUMMARY_PATH.exists(), (
        "Robustness summary is missing. "
        "Run notebooks/05_robustness.ipynb first."
    )

    return pd.read_csv(
        ROBUSTNESS_SUMMARY_PATH
    )


def load_overlap_summary() -> dict[str, float]:
    assert OVERLAP_SUMMARY_PATH.exists(), (
        "Overlap diagnostics are missing. "
        "Run notebooks/05_robustness.ipynb first."
    )

    overlap_df = pd.read_csv(
        OVERLAP_SUMMARY_PATH
    )

    return dict(
        zip(
            overlap_df["metric"],
            overlap_df["value"],
        )
    )


def get_analysis_row(
    summary_df: pd.DataFrame,
    analysis_name: str,
) -> pd.Series:
    matching_rows = summary_df.loc[
        summary_df["analysis"] == analysis_name
    ]

    assert len(matching_rows) == 1

    return matching_rows.iloc[0]


def test_robustness_output_files_exist() -> None:
    assert ROBUSTNESS_SUMMARY_PATH.exists()
    assert OVERLAP_SUMMARY_PATH.exists()


def test_placebo_effect_is_small() -> None:
    summary_df = load_robustness_summary()

    placebo_row = get_analysis_row(
        summary_df,
        "Placebo Treatment",
    )

    assert abs(placebo_row["estimate"]) < (
        2.5 * placebo_row["standard_error"]
    )


def test_reference_interval_contains_true_effect() -> None:
    summary_df = load_robustness_summary()

    reference_row = get_analysis_row(
        summary_df,
        "Reference AIPW",
    )

    assert (
        reference_row["ci_lower_95"]
        <= reference_row["target_effect"]
        <= reference_row["ci_upper_95"]
    )


def test_omitted_confounder_result_is_finite() -> None:
    summary_df = load_robustness_summary()

    reduced_row = get_analysis_row(
        summary_df,
        "Omitted Confounders",
    )

    checked_values = reduced_row[
        [
            "estimate",
            "standard_error",
            "ci_lower_95",
            "ci_upper_95",
            "signed_error",
            "absolute_error",
        ]
    ].to_numpy(dtype=float)

    assert np.isfinite(
        checked_values
    ).all()


def test_empirical_overlap_is_adequate() -> None:
    overlap = load_overlap_summary()

    assert overlap["overlap_fraction"] >= 0.99


def test_no_extreme_propensities() -> None:
    overlap = load_overlap_summary()

    assert overlap[
        "extreme_propensity_count"
    ] == 0

    