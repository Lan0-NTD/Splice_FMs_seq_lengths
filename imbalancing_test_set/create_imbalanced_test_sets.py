from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd


SEED = 42
WINDOWS = [300, 600, 1000, 2000, 10000]
RATIOS = [10, 20, 50, 100]  # donor:acceptor:negative = 1:1:R
CHROMS = {"chr20", "chr21"}


def detect_label_mapping(values: Iterable[object]) -> Dict[str, object]:
    """Detect raw values used for donor/acceptor/negative in Splicing_types."""
    donor_candidates = ["1", "donor", "d"]
    acceptor_candidates = ["2", "acceptor", "acceptor_site", "acc", "a"]
    negative_candidates = ["0", "negative", "neg", "none", "other", "n"]

    normalized_to_raw: Dict[str, object] = {}
    for raw_value in values:
        normalized = str(raw_value).strip().lower()
        normalized_to_raw[normalized] = raw_value

    def pick(candidates: Iterable[str], label_name: str) -> object:
        for candidate in candidates:
            if candidate in normalized_to_raw:
                return normalized_to_raw[candidate]
        unique_values = ", ".join(sorted(normalized_to_raw.keys()))
        raise ValueError(
            f"Khong tim thay nhan '{label_name}' trong Splicing_types. "
            f"Cac gia tri hien co: [{unique_values}]"
        )

    return {
        "donor": pick(donor_candidates, "donor"),
        "acceptor": pick(acceptor_candidates, "acceptor"),
        "negative": pick(negative_candidates, "negative"),
    }


def sample_ratio_dataframe(df: pd.DataFrame, ratio: int, seed: int) -> Tuple[pd.DataFrame, Dict[str, int]]:
    mapping = detect_label_mapping(df["Splicing_types"].dropna().unique())

    donor_df = df[df["Splicing_types"] == mapping["donor"]]
    acceptor_df = df[df["Splicing_types"] == mapping["acceptor"]]
    negative_df = df[df["Splicing_types"] == mapping["negative"]]

    max_pos_by_negative = len(negative_df) // ratio
    n_pos = min(len(donor_df), len(acceptor_df), max_pos_by_negative)

    if n_pos <= 0:
        raise ValueError(
            "Khong the tao tap du lieu voi ratio nay vi so mau khong du. "
            f"donor={len(donor_df)}, acceptor={len(acceptor_df)}, negative={len(negative_df)}, ratio={ratio}"
        )

    n_neg = n_pos * ratio

    donor_sample = donor_df.sample(n=n_pos, random_state=seed, replace=False)
    acceptor_sample = acceptor_df.sample(n=n_pos, random_state=seed, replace=False)
    negative_sample = negative_df.sample(n=n_neg, random_state=seed, replace=False)

    sampled_df = pd.concat([donor_sample, acceptor_sample, negative_sample], axis=0)
    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    stats = {
        "donor": len(donor_sample),
        "acceptor": len(acceptor_sample),
        "negative": len(negative_sample),
    }
    return sampled_df, stats


def process_dataset(
    source_name: str,
    source_dir: Path,
    output_dir: Path,
    windows: Iterable[int],
    ratios: Iterable[int],
    seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for window in windows:
        input_file = source_dir / f"{source_name}{window}.csv"
        if not input_file.exists():
            print(f"[SKIP] Khong tim thay file: {input_file}")
            continue

        print("=" * 100)
        print(f"Dang xu ly: {input_file}")

        df = pd.read_csv(input_file)
        required_cols = {"Splicing_types", "CHROM"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise ValueError(f"File {input_file} thieu cot bat buoc: {sorted(missing_cols)}")

        test_df = df[df["CHROM"].isin(CHROMS)].copy()
        print(f"Tong so mau test (chr20/chr21): {len(test_df):,}")
        print("value_counts Splicing_types (tap test truoc subsample):")
        print(test_df["Splicing_types"].value_counts(dropna=False))
        print()

        for ratio in ratios:
            sampled_df, stats = sample_ratio_dataframe(test_df, ratio=ratio, seed=seed)
            ratio_suffix = f"1_1_{ratio}"
            output_file = output_dir / f"{source_name}{window}_test_set_{ratio_suffix}.csv"
            sampled_df.to_csv(output_file, index=False)

            print(f"[DONE] {output_file.name}")
            print(f"Kich thuoc: {len(sampled_df):,} | donor={stats['donor']:,}, acceptor={stats['acceptor']:,}, negative={stats['negative']:,}")
            print("value_counts Splicing_types (sau subsample):")
            print(sampled_df["Splicing_types"].value_counts(dropna=False))
            print("-" * 100)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    process_dataset(
        source_name="gencode",
        source_dir=project_root / "gencode_multi_seq_length",
        output_dir=project_root / "imbalancing_test_set" / "gencode",
        windows=WINDOWS,
        ratios=RATIOS,
        seed=SEED,
    )

    process_dataset(
        source_name="gtex",
        source_dir=project_root / "gtex_multi_seq_length",
        output_dir=project_root / "imbalancing_test_set" / "gtex",
        windows=WINDOWS,
        ratios=RATIOS,
        seed=SEED,
    )

    print("Hoan tat tao cac tap test imbalanced.")


if __name__ == "__main__":
    main()
