from datetime import datetime
from pathlib import Path
import gc
import os
import time

import numpy as np
import pandas as pd
import torch


def run(context: dict):
    config = context["config"]
    extractor = context["extractor"]
    EMBEDDINGS_DIR = context["EMBEDDINGS_DIR"]
    EMBEDDING_CONFIG = context["EMBEDDING_CONFIG"]
    MODELS_CONFIG = context["MODELS_CONFIG"]
    WINDOWS = context["WINDOWS"]
    IMBALANCED_RATIOS = context["IMBALANCED_RATIOS"]
    IMBALANCED_SOURCE_DIRS = context["IMBALANCED_SOURCE_DIRS"]
    RAW_RATIO_TAG = context["RAW_RATIO_TAG"]
    TELEMETRY_DIR = context["TELEMETRY_DIR"]
    device = context["device"]
    psutil = context.get("psutil")
    _build_raw_default_csv = context["_build_raw_default_csv"]
    _param_stats = context["_param_stats"]
    _measure_fm_gflops_per_sample = context["_measure_fm_gflops_per_sample"]

    DNABERT_POOLING_METHOD = "center"

    def _norm_path_str(p):
        return os.path.normcase(os.path.normpath(str(Path(p)).replace("/", "\\")))

    def _dnabert_max_length(tokenizer, window_size: int):
        tok_max = getattr(tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 0 < tok_max < 100000:
            return min(EMBEDDING_CONFIG.get("max_length", window_size), tok_max)
        return min(EMBEDDING_CONFIG.get("max_length", window_size), 512)

    def _extract_one_csv_center(model, tokenizer, csv_path, output_path, window_size, source_name, ratio, model_name, model_id, fm_param_count, fm_param_memory_mb, fm_gflops_per_sample, fm_gflops_method, rows):
        df = pd.read_csv(csv_path)
        if output_path.exists():
            output_path.unlink()

        max_length = _dnabert_max_length(tokenizer, window_size)
        batch_size = EMBEDDING_CONFIG.get("batch_size_by_window", {}).get(window_size, EMBEDDING_CONFIG.get("batch_size", 64))

        cpu_before_mb = np.nan
        cpu_after_mb = np.nan
        if psutil is not None:
            proc = psutil.Process()
            cpu_before_mb = proc.memory_info().rss / (1024 ** 2)

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        start_time = time.perf_counter()
        ok = extractor._extract_with_fallback(
            sequences=df["sequence"].values,
            labels=df["Splicing_types"].values,
            model=model,
            tokenizer=tokenizer,
            output_file=output_path,
            max_length=max_length,
            batch_size=batch_size,
            method=DNABERT_POOLING_METHOD,
            use_fp16_current=False,
        )
        elapsed = float(time.perf_counter() - start_time)
        if not ok:
            raise RuntimeError(f"Extract failed for {csv_path.name}")

        gpu_peak_mb = np.nan
        if device == "cuda" and torch.cuda.is_available():
            gpu_peak_mb = float(torch.cuda.max_memory_allocated() / (1024 ** 2))

        if psutil is not None:
            proc = psutil.Process()
            cpu_after_mb = proc.memory_info().rss / (1024 ** 2)

        rows.append(
            {
                "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                "stage": "extract_embedding",
                "family": model_name,
                "model_id": model_id,
                "model_variant": model_id,
                "window_size": int(window_size),
                "dataset": source_name,
                "ratio": ratio,
                "input_csv": str(csv_path),
                "output_embedding": str(output_path),
                "n_samples": int(len(df)),
                "running_time_seconds": elapsed,
                "param_count": int(fm_param_count),
                "model_param_memory_mb": float(fm_param_memory_mb),
                "gpu_peak_memory_mb": float(gpu_peak_mb),
                "cpu_ram_delta_mb": float(cpu_after_mb - cpu_before_mb) if psutil is not None else np.nan,
                "gflops_per_sample": float(fm_gflops_per_sample),
                "gflops_total": float(fm_gflops_per_sample * len(df)),
                "gflops_method": fm_gflops_method,
                "batch_size": int(batch_size),
                "max_length": int(max_length),
                "pooling_method": DNABERT_POOLING_METHOD,
            }
        )

    dnabert_rows = []
    for window_size in WINDOWS:
        raw_csv_by_source = {
            "gencode": _build_raw_default_csv(window_size, "gencode"),
            "gtex": _build_raw_default_csv(window_size, "gtex"),
        }

        model_name = "DNABert"
        for model_id in MODELS_CONFIG[model_name].get("model_ids", []):
            skip_reason = config.get_model_window_skip_reason(model_name, model_id, window_size)
            if skip_reason is not None:
                print(f"[SKIP] {skip_reason}")
                continue

            combo_name = f"{model_name}_{model_id}"
            model_dir = EMBEDDINGS_DIR / str(window_size) / combo_name
            model_dir.mkdir(parents=True, exist_ok=True)

            print(f"[LOAD] {combo_name} | window={window_size} | pooling={DNABERT_POOLING_METHOD}")
            model, tokenizer = extractor.model_loader.load_model_by_name(model_name, model_id)
            fm_param_count, fm_param_memory_mb = _param_stats(model)
            probe_seq = "ACGT" * max(50, min(window_size // 4, 500))
            fm_gflops_per_sample, fm_gflops_method = _measure_fm_gflops_per_sample(
                model=model, tokenizer=tokenizer, sample_seq=probe_seq, max_length=window_size
            )

            for source_name in ["gencode", "gtex"]:
                raw_csv = raw_csv_by_source[source_name]
                raw_out = model_dir / ("test_embeddings.pt" if source_name == "gencode" else "gtex_test_embeddings.pt")
                _extract_one_csv_center(
                    model,
                    tokenizer,
                    raw_csv,
                    raw_out,
                    window_size,
                    source_name,
                    RAW_RATIO_TAG,
                    model_name,
                    model_id,
                    fm_param_count,
                    fm_param_memory_mb,
                    fm_gflops_per_sample,
                    fm_gflops_method,
                    dnabert_rows,
                )

            for source_name, source_dir in IMBALANCED_SOURCE_DIRS.items():
                for ratio in IMBALANCED_RATIOS:
                    csv_file = source_dir / f"{source_name}{window_size}_test_set_{ratio}.csv"
                    if not csv_file.exists():
                        continue
                    out_file = model_dir / (
                        f"test_{ratio}_embeddings.pt" if source_name == "gencode" else f"gtex_test_{ratio}_embeddings.pt"
                    )
                    _extract_one_csv_center(
                        model,
                        tokenizer,
                        csv_file,
                        out_file,
                        window_size,
                        source_name,
                        ratio,
                        model_name,
                        model_id,
                        fm_param_count,
                        fm_param_memory_mb,
                        fm_gflops_per_sample,
                        fm_gflops_method,
                        dnabert_rows,
                    )

            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not dnabert_rows:
        raise RuntimeError("Khong tao duoc DNABERT telemetry nao.")

    dnabert_df = pd.DataFrame(dnabert_rows)
    new_keys = set(dnabert_df["output_embedding"].map(_norm_path_str))

    extract_candidates = sorted(TELEMETRY_DIR.glob("extract_telemetry_long_*.csv"))
    if not extract_candidates:
        raise RuntimeError("Khong tim thay extract_telemetry_long de merge")

    base_extract = pd.read_csv(extract_candidates[-1])
    base_extract["__key"] = base_extract["output_embedding"].map(_norm_path_str)
    base_keep = base_extract[~base_extract["__key"].isin(new_keys)].drop(columns="__key")

    merged_extract_df = pd.concat([base_keep, dnabert_df], ignore_index=True)
    merged_extract_df = merged_extract_df.sort_values(["window_size", "family", "model_id", "dataset", "ratio"]).reset_index(drop=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_long_csv = TELEMETRY_DIR / f"extract_telemetry_long_{ts}.csv"
    merged_long_json = TELEMETRY_DIR / f"extract_telemetry_long_{ts}.json"
    merged_extract_df.to_csv(merged_long_csv, index=False)
    merged_extract_df.to_json(merged_long_json, orient="records", indent=2)

    print("[DONE] DNABERT center re-extract + full telemetry merge")
    print("[CHECK] pooling_method counts in DNABERT rows:")
    print(dnabert_df["pooling_method"].value_counts(dropna=False))
    print(f"[SAVED] {merged_long_csv.name} | rows={len(merged_extract_df)}")
