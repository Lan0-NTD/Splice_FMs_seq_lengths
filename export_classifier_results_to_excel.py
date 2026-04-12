from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
CLASSIFIERS_DIR = ROOT_DIR / "results" / "classifiers"
DEFAULT_OUTPUT = CLASSIFIERS_DIR / "summaries" / "classifier_results_summary.xlsx"

WINDOW_PATTERN = re.compile(r"_w(\d+)$")
PRIMARY_SUMMARY_COLUMNS = [
    "model_family",
    "model_name",
    "model_key",
    "window_size",
    "embedding_dim",
    "num_folds",
    "best_fold_number",
    "best_fold_mcc",
    "avg_accuracy",
    "avg_balanced_accuracy",
    "avg_f1_weighted",
    "avg_f1_macro",
    "avg_mcc",
    "avg_pr_auc_macro",
    "avg_roc_auc_macro",
]
PIVOT_METRICS = {
    "MCC": "avg_mcc",
    "Accuracy": "avg_accuracy",
    "F1_Weighted": "avg_f1_weighted",
    "PR_AUC_Macro": "avg_pr_auc_macro",
    "ROC_AUC_Macro": "avg_roc_auc_macro",
}

PR_AUC_CLASS_COLUMNS = ["pr_auc_Other", "pr_auc_Donor", "pr_auc_Acceptor"]


def is_excluded_result(result_path: Path) -> bool:
    experiment_dir = result_path.parent.name
    return "CLS" in experiment_dir


def parse_window_size(experiment_dir: str) -> int:
    match = WINDOW_PATTERN.search(experiment_dir)
    if not match:
        raise ValueError(f"Unable to parse window size from experiment directory: {experiment_dir}")
    return int(match.group(1))


def parse_model_name(experiment_dir: str) -> str:
    return WINDOW_PATTERN.sub("", experiment_dir)


def collect_numeric_metric_means(per_fold_results: dict) -> dict[str, float]:
    metric_values: dict[str, list[float]] = {}

    for fold_data in per_fold_results.values():
        metrics = fold_data.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                metric_values.setdefault(metric_name, []).append(float(metric_value))

        pr_auc_values = [metrics[column_name] for column_name in PR_AUC_CLASS_COLUMNS if isinstance(metrics.get(column_name), (int, float))]
        if len(pr_auc_values) == len(PR_AUC_CLASS_COLUMNS):
            metric_values.setdefault("pr_auc_macro", []).append(mean(pr_auc_values))

    return {
        f"avg_{metric_name}": mean(values)
        for metric_name, values in metric_values.items()
        if values
    }


def build_summary_row(result_path: Path, payload: dict) -> dict[str, object]:
    relative_parts = result_path.relative_to(CLASSIFIERS_DIR).parts
    model_family = relative_parts[0]
    experiment_dir = relative_parts[1]
    model_name = parse_model_name(experiment_dir)
    window_size = parse_window_size(experiment_dir)
    best_fold = payload.get("best_fold", {})
    per_fold_results = payload.get("per_fold_results", {})

    row = {
        "model_family": model_family,
        "model_name": model_name,
        "model_key": f"{model_family}/{model_name}",
        "window_size": window_size,
        "experiment_name": payload.get("experiment_name", ""),
        "experiment_dir": experiment_dir,
        "source_json": str(result_path),
        "timestamp": payload.get("timestamp", ""),
        "embedding_dim": payload.get("embedding_dim"),
        "num_folds": payload.get("num_folds"),
        "best_fold_number": best_fold.get("fold_number"),
        "best_fold_index": best_fold.get("fold_idx"),
        "best_fold_metric": best_fold.get("best_metric"),
        "best_fold_mcc": best_fold.get("best_mcc"),
        "best_checkpoint": best_fold.get("checkpoint"),
        "best_checkpoint_path": best_fold.get("checkpoint_path"),
    }
    row.update(collect_numeric_metric_means(per_fold_results))
    return row


def build_per_fold_rows(result_path: Path, payload: dict) -> list[dict[str, object]]:
    relative_parts = result_path.relative_to(CLASSIFIERS_DIR).parts
    model_family = relative_parts[0]
    experiment_dir = relative_parts[1]
    model_name = parse_model_name(experiment_dir)
    window_size = parse_window_size(experiment_dir)
    rows: list[dict[str, object]] = []

    for fold_key, fold_data in payload.get("per_fold_results", {}).items():
        row = {
            "model_family": model_family,
            "model_name": model_name,
            "model_key": f"{model_family}/{model_name}",
            "window_size": window_size,
            "experiment_name": payload.get("experiment_name", ""),
            "source_json": str(result_path),
            "fold_key": fold_key,
            "best_epoch": fold_data.get("best_epoch"),
            "best_metric": fold_data.get("best_metric"),
            "best_mcc": fold_data.get("best_mcc"),
            "best_f1_at_best_epoch": fold_data.get("best_f1_at_best_epoch"),
            "best_checkpoint": fold_data.get("best_checkpoint"),
            "val_loss": fold_data.get("val_loss"),
        }

        metrics = fold_data.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                row[metric_name] = float(metric_value)

        pr_auc_values = [metrics[column_name] for column_name in PR_AUC_CLASS_COLUMNS if isinstance(metrics.get(column_name), (int, float))]
        if len(pr_auc_values) == len(PR_AUC_CLASS_COLUMNS):
            row["pr_auc_macro"] = mean(pr_auc_values)

        rows.append(row)

    return rows


def autosize_worksheet(worksheet) -> None:
    for column_cells in worksheet.columns:
        values = [str(cell.value) for cell in column_cells if cell.value is not None]
        if not values:
            continue
        max_length = max(len(value) for value in values)
        column_letter = column_cells[0].column_letter
        worksheet.column_dimensions[column_letter].width = min(max(max_length + 2, 12), 60)


def write_workbook(summary_df: pd.DataFrame, per_fold_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="All_Experiments", index=False)
        per_fold_df.to_excel(writer, sheet_name="Per_Fold", index=False)

        for sheet_name, metric_column in PIVOT_METRICS.items():
            pivot_df = (
                summary_df.pivot(index="model_key", columns="window_size", values=metric_column)
                .sort_index()
                .sort_index(axis=1)
            )
            pivot_df.to_excel(writer, sheet_name=sheet_name)

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            autosize_worksheet(worksheet)


def main() -> None:
    result_files = sorted(CLASSIFIERS_DIR.glob("*/*/results.json"))
    included_files = [path for path in result_files if not is_excluded_result(path)]

    summary_rows: list[dict[str, object]] = []
    per_fold_rows: list[dict[str, object]] = []

    for result_path in included_files:
        with result_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        summary_rows.append(build_summary_row(result_path, payload))
        per_fold_rows.extend(build_per_fold_rows(result_path, payload))

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("No classifier results were found.")

    ordered_columns = [
        column_name
        for column_name in PRIMARY_SUMMARY_COLUMNS
        if column_name in summary_df.columns
    ]
    ordered_columns.extend(
        sorted(column_name for column_name in summary_df.columns if column_name not in ordered_columns)
    )
    summary_df = summary_df[ordered_columns].sort_values(
        by=["model_family", "model_name", "window_size"],
        ascending=[True, True, True],
    )

    per_fold_df = pd.DataFrame(per_fold_rows).sort_values(
        by=["model_family", "model_name", "window_size", "fold_key"],
        ascending=[True, True, True, True],
    )

    write_workbook(summary_df, per_fold_df, DEFAULT_OUTPUT)

    print(f"Created workbook: {DEFAULT_OUTPUT}")
    print(f"Included experiments: {len(summary_df)}")
    print(f"Included per-fold rows: {len(per_fold_df)}")


if __name__ == "__main__":
    main()