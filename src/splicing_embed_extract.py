"""
Standalone script to extract embeddings from foundation models (offline)

Usage:
    python src/splicing_embed_extract.py
    
Output:
    data/embeddings/{window_size}/{model_name}/
    ├── train_embeddings.pt    [N_train, embedding_dim]
    ├── val_embeddings.pt      [N_val, embedding_dim]
    └── test_embeddings.pt     [N_test, embedding_dim]
"""

import sys
import inspect
import traceback
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
project_dir = Path(__file__).parent.parent
sys.path.insert(0, str(project_dir / "src"))

from config import *
from models import FoundationModelLoader
from data_preparation import DNADataPreparation


class EmbeddingExtractor:
    """Extract embeddings from foundation models and save to disk"""
    
    def __init__(self, device='cuda'):
        if device == 'cuda' and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = 'cpu'

        self.device = device
        self.model_loader = FoundationModelLoader(device=device)
        self.data_prep = DNADataPreparation(
            raw_data_dir=RAW_DATA_DIR,
            gtex_data_dir=GTEX_DATA_DIR,
            processed_data_dir=PROCESSED_DATA_DIR,
            window_sizes=WINDOW_SIZES
        )
        self.embed_dir = EMBEDDINGS_DIR
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self._configure_safe_attention_backend()

    def _configure_safe_attention_backend(self):
        """Force safe math attention backend to avoid Triton/flash kernel incompatibilities."""
        if not torch.cuda.is_available():
            return

        try:
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_flash_sdp"):
                torch.backends.cuda.enable_flash_sdp(False)
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
                torch.backends.cuda.enable_mem_efficient_sdp(False)
            if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "enable_math_sdp"):
                torch.backends.cuda.enable_math_sdp(True)
            logger.info("Configured safe attention backend: flash_sdp=False, mem_efficient_sdp=False, math_sdp=True")
        except Exception as e:
            logger.warning(f"Could not configure safe attention backend: {e}")
    
    def extract_embeddings_batch(self, sequences, model, tokenizer, max_length=512, batch_size=256, method='center', use_fp16_override=None, run_device=None):
        """
        Extract embeddings from sequences in batches (Optimized version)
        """
        all_embeddings = []
        model.eval()
        
        # Get optimization settings from config
        use_fp16 = EMBEDDING_CONFIG.get('use_fp16', False)
        if use_fp16_override is not None:
            use_fp16 = use_fp16_override
        effective_device = run_device or self.device
        
        logger.info(f"Extracting embeddings using '{method}' method for {len(sequences)} sequences")
        logger.info(f"Batch size: {batch_size}, FP16: {use_fp16}")

        try:
            model_forward_params = set(inspect.signature(model.forward).parameters.keys())
        except Exception:
            model_forward_params = {"input_ids", "attention_mask", "output_hidden_states"}

        supports_attention_mask = "attention_mask" in model_forward_params
        supports_output_hidden_states = "output_hidden_states" in model_forward_params
        
        # Sử dụng inference_mode thay cho no_grad để tối ưu tốc độ hơn nữa
        with torch.inference_mode():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting embeddings"):
                batch_seqs = sequences[i:i+batch_size]
                
                # Convert numpy array to list of strings for tokenizer
                if hasattr(batch_seqs, 'tolist'):
                    batch_seqs = batch_seqs.tolist()
                
                # Tối ưu: Dùng dynamic padding (padding=True) thay vì max_length cố định
                encodings = tokenizer(
                    batch_seqs,
                    return_tensors="pt",
                    padding=True,  
                    truncation=True,
                    max_length=max_length,
                    return_attention_mask=True,
                )
                
                input_ids_cpu = encodings["input_ids"]
                attention_mask_cpu = encodings["attention_mask"]

                if 'cuda' in effective_device and torch.cuda.is_available():
                    input_ids = input_ids_cpu.pin_memory().to(effective_device, non_blocking=True)
                    attention_mask = attention_mask_cpu.pin_memory().to(effective_device, non_blocking=True)
                else:
                    input_ids = input_ids_cpu.to(effective_device)
                    attention_mask = attention_mask_cpu.to(effective_device)

                model_inputs = {
                    "input_ids": input_ids,
                }
                if supports_attention_mask:
                    model_inputs["attention_mask"] = attention_mask
                if supports_output_hidden_states:
                    model_inputs["output_hidden_states"] = True
                
                # Cập nhật syntax autocast mới nhất của PyTorch
                if use_fp16:
                    with torch.amp.autocast(device_type='cuda' if 'cuda' in effective_device else 'cpu'):
                        outputs = model(**model_inputs)
                else:
                    outputs = model(**model_inputs)
                
                if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                    hidden_states = outputs.last_hidden_state
                elif isinstance(outputs, (tuple, list)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
                    hidden_states = outputs[0]
                elif isinstance(outputs, dict):
                    if "last_hidden_state" in outputs and torch.is_tensor(outputs["last_hidden_state"]):
                        hidden_states = outputs["last_hidden_state"]
                    elif "hidden_states" in outputs and outputs["hidden_states"]:
                        hidden_states = outputs["hidden_states"][-1]
                    else:
                        raise RuntimeError("Model output does not contain hidden states")
                else:
                    raise RuntimeError("Unsupported model output format")

                if isinstance(hidden_states, (tuple, list)) and len(hidden_states) > 0:
                    hidden_states = hidden_states[-1]

                batch_size_current = hidden_states.shape[0]
                
                # TỐI ƯU: Vector hóa toàn bộ quá trình trích xuất (Không dùng vòng lặp for)
                if method == 'center':
                    seq_lens = attention_mask.sum(dim=1)  # [batch_size]
                    center_idxs = seq_lens // 2           # [batch_size]
                    batch_idxs = torch.arange(batch_size_current, device=effective_device)
                    batch_embeddings = hidden_states[batch_idxs, center_idxs, :]
                    
                elif method == 'mean':
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                    sum_embeddings = torch.sum(hidden_states * input_mask_expanded, 1)
                    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                    batch_embeddings = sum_embeddings / sum_mask
                    
                elif method == 'cls':
                    batch_embeddings = hidden_states[:, 0, :]
                    
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Đưa tensor về CPU ngay lập tức để giải phóng VRAM, thêm vào list
                all_embeddings.append(batch_embeddings.cpu())
                
                # LƯU Ý: Đã bỏ torch.cuda.empty_cache() ở đây để tránh làm chậm tốc độ
        
        # Ghép tất cả các tensor trong list thành 1 tensor duy nhất thay vì dùng stack
        return torch.cat(all_embeddings, dim=0)

    def _extract_with_fallback(self, sequences, labels, model, tokenizer, output_file, max_length, batch_size, method, use_fp16_current):
        """Extract embeddings with automatic fallback to CPU on CUDA/Triton failures."""
        try:
            embeddings = self.extract_embeddings_batch(
                sequences,
                model,
                tokenizer,
                max_length=max_length,
                batch_size=batch_size,
                method=method,
                use_fp16_override=use_fp16_current,
                run_device=self.device,
            )
            self.save_embeddings(embeddings, labels, output_file)
            return True
        except Exception as e:
            error_text = str(e)
            error_text_lower = error_text.lower()
            is_triton_kernel_error = (
                "dot() got an unexpected keyword argument 'trans_b'" in error_text
                or "tl.dot" in error_text
            )
            is_cuda_runtime_error = (
                "device-side assert triggered" in error_text_lower
                or "cuda error" in error_text_lower
                or "acceleratorerror" in error_text_lower
                or "cudnn" in error_text_lower
            )

            should_cpu_fallback = (is_triton_kernel_error or is_cuda_runtime_error)

            if should_cpu_fallback and 'cuda' in self.device:
                logger.warning("Detected CUDA/Triton runtime issue. Retrying extraction on CPU...")
                try:
                    try:
                        model = model.to('cpu')
                    except Exception as move_err:
                        logger.warning(f"Could not move model to CPU before fallback: {move_err}")
                    try:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        # CUDA context can be poisoned after device-side assert; ignore and continue on CPU path.
                        pass

                    embeddings = self.extract_embeddings_batch(
                        sequences,
                        model,
                        tokenizer,
                        max_length=max_length,
                        batch_size=max(8, min(batch_size, 64)),
                        method=method,
                        use_fp16_override=False,
                        run_device='cpu',
                    )
                    self.save_embeddings(embeddings, labels, output_file)
                    logger.info("✓ CPU fallback extraction succeeded")

                    # Move model back to original device for next tasks (best effort)
                    try:
                        model.to(self.device)
                    except Exception:
                        pass
                    return True
                except Exception as cpu_e:
                    logger.error(f"✗ CPU fallback extraction also failed: {type(cpu_e).__name__}: {repr(cpu_e)}")
                    logger.error(traceback.format_exc())
                    return False

            logger.error(f"✗ Embedding extraction failed without fallback match: {type(e).__name__}: {repr(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def save_embeddings(self, embeddings, labels, output_path):
        """Save embeddings and labels to .pt file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data_dict = {
            "embeddings": embeddings.cpu().detach(),
            "labels": torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels.cpu().detach()
        }
        
        torch.save(data_dict, output_path)
        logger.info(f"✓ Saved embeddings: {output_path}")
        logger.info(f"  Shape: embeddings={data_dict['embeddings'].shape}, labels={data_dict['labels'].shape}")
    
    def extract_for_window_and_model(self, window_size, model_name, model_id, output_suffix=""):
        """Extract embeddings for one window size and one model"""

        skip_reason = get_model_window_skip_reason(model_name, model_id, window_size)
        if skip_reason is not None:
            logger.info(skip_reason)
            return 'unsupported'
        
        logger.info(f"\n{'='*70}")
        combo_name = f"{model_name}_{model_id}{output_suffix}"
        logger.info(f"Extracting: {combo_name} | Window Size: {window_size}")
        logger.info(f"{'='*70}")
        
        # Get adaptive batch size based on window size
        batch_size_map = EMBEDDING_CONFIG.get('batch_size_by_window', {})
        adaptive_batch_size = batch_size_map.get(window_size, EMBEDDING_CONFIG['batch_size'])
        max_length = EMBEDDING_CONFIG.get('max_length', window_size)
        logger.info(f"Using batch size: {adaptive_batch_size} (for window {window_size})")
        logger.info(f"Using max_length: {max_length}")
        
        # Create output directory
        model_dir = self.embed_dir / str(window_size) / combo_name
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if all embeddings already exist
        trainval_file = model_dir / "trainval_embeddings.pt"
        test_file = model_dir / "test_embeddings.pt"
        gtex_file = model_dir / "gtex_test_embeddings.pt"
        
        all_exist = trainval_file.exists() and test_file.exists() and gtex_file.exists()
        
        if all_exist:
            logger.info(f"✓ All embeddings already exist, skipping...")
            logger.info(f"  {trainval_file.name} ✓")
            logger.info(f"  {test_file.name} ✓")
            logger.info(f"  {gtex_file.name} ✓")
            return 'skipped'  # Return 'skipped' instead of True
        
        # Load model
        try:
            logger.info(f"Loading model: {model_name}/{model_id}...")
            model, tokenizer = self.model_loader.load_model_by_name(model_name, model_id)
            logger.info(f"✓ Model loaded")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            return False

        # Use a safe per-model max_length (DNABERT is typically much shorter than 10k)
        effective_max_length = max_length
        extraction_method = 'center'
        if model_name == "DNABert":
            tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
            if isinstance(tokenizer_max_len, int) and 0 < tokenizer_max_len < 100000:
                effective_max_length = min(max_length, tokenizer_max_len)
            else:
                effective_max_length = min(max_length, 512)
            logger.info(f"DNABERT effective max_length: {effective_max_length}")
            extraction_method = 'center'
        logger.info(f"Embedding pooling method for {model_name}: {extraction_method}")
        
        # Load and split gencode data
        try:
            logger.info(f"Loading GENCODE data for window size {window_size}...")
            gencode_df = self.data_prep.load_gencode_data(window_size)
            train_df, val_df, test_df = self.data_prep.split_by_chromosome(gencode_df)
            trainval_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)
            logger.info(f"✓ GENCODE loaded:")
            logger.info(f"  Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
            logger.info(f"  Train+Val (for CV): {len(trainval_df):,}")
        except Exception as e:
            logger.error(f"✗ Failed to load GENCODE data: {e}")
            return False
        
        # Extract train+val embeddings (for CV)
        try:
            logger.info(f"\nExtracting GENCODE TRAIN+VAL embeddings (for CV)...")
            trainval_seqs = trainval_df['sequence'].values
            trainval_labels = trainval_df['Splicing_types'].values
            use_fp16_current = EMBEDDING_CONFIG.get('use_fp16', False)
            if model_name == "DNABert":
                use_fp16_current = False
            
            ok = self._extract_with_fallback(
                sequences=trainval_seqs,
                labels=trainval_labels,
                model=model,
                tokenizer=tokenizer,
                output_file=model_dir / "trainval_embeddings.pt",
                max_length=effective_max_length,
                batch_size=adaptive_batch_size,
                method=extraction_method,
                use_fp16_current=use_fp16_current,
            )
            if not ok:
                return False
        except Exception as e:
            logger.error(f"✗ Failed to extract train+val embeddings: {e}")
            return False
        
        # Extract test embeddings (GENCODE)
        try:
            logger.info(f"\nExtracting GENCODE TEST embeddings...")
            test_seqs = test_df['sequence'].values
            test_labels = test_df['Splicing_types'].values
            use_fp16_current = EMBEDDING_CONFIG.get('use_fp16', False)
            if model_name == "DNABert":
                use_fp16_current = False
            
            ok = self._extract_with_fallback(
                sequences=test_seqs,
                labels=test_labels,
                model=model,
                tokenizer=tokenizer,
                output_file=model_dir / "test_embeddings.pt",
                max_length=effective_max_length,
                batch_size=adaptive_batch_size,
                method=extraction_method,
                use_fp16_current=use_fp16_current,
            )
            if not ok:
                return False
        except Exception as e:
            logger.error(f"✗ Failed to extract test embeddings: {e}")
            return False
        
        # Load and extract GTEx data
        try:
            logger.info(f"\nLoading GTEx data for window size {window_size}...")
            gtex_df = self.data_prep.load_gtex_data(window_size)
            logger.info(f"✓ GTEx loaded: {len(gtex_df):,} samples")
            
            gtex_seqs = gtex_df['sequence'].values
            gtex_labels = gtex_df['Splicing_types'].values
            use_fp16_current = EMBEDDING_CONFIG.get('use_fp16', False)
            if model_name == "DNABert":
                use_fp16_current = False
            
            logger.info(f"Extracting GTEx TEST embeddings...")
            ok = self._extract_with_fallback(
                sequences=gtex_seqs,
                labels=gtex_labels,
                model=model,
                tokenizer=tokenizer,
                output_file=model_dir / "gtex_test_embeddings.pt",
                max_length=effective_max_length,
                batch_size=adaptive_batch_size,
                method=extraction_method,
                use_fp16_current=use_fp16_current,
            )
            if not ok:
                return False
        except Exception as e:
            logger.error(f"✗ Failed to extract GTEx embeddings: {e}")
            return False
        
        logger.info(f"\n✓ Successfully extracted all embeddings for {combo_name} (window {window_size})")

        import gc
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True
    
    def extract_all(self, window_sizes=None, models_config=None, output_suffix=""):
        """Extract embeddings for all combinations"""
        
        if window_sizes is None:
            window_sizes = WINDOW_SIZES
        
        if models_config is None:
            models_config = MODELS_CONFIG
        
        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING EXTRACTION - START")
        logger.info(f"{'='*80}")
        logger.info(f"Window sizes: {window_sizes}")
        logger.info(f"Models: {list(models_config.keys())}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {self.embed_dir}")
        
        stats = {
            'total_started': 0,
            'total_succeeded': 0,
            'total_failed': 0,
            'total_skipped': 0,
            'total_skipped_existing': 0,
            'total_skipped_unsupported': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        for window_size in window_sizes:
            logger.info(f"\n{'#'*80}")
            logger.info(f"WINDOW SIZE: {window_size}")
            logger.info(f"{'#'*80}")
            
            for model_name, config in models_config.items():
                for model_id in config['model_ids']:
                    stats['total_started'] += 1
                    
                    try:
                        result = self.extract_for_window_and_model(
                            window_size,
                            model_name,
                            model_id,
                            output_suffix=output_suffix,
                        )
                        
                        if result == 'skipped':
                            stats['total_skipped'] += 1
                            stats['total_skipped_existing'] += 1
                        elif result == 'unsupported':
                            stats['total_skipped'] += 1
                            stats['total_skipped_unsupported'] += 1
                        elif result is True:
                            stats['total_succeeded'] += 1
                        else:
                            stats['total_failed'] += 1
                            stats['errors'].append(f"{model_name}_{model_id}_ws{window_size}: extraction failed")
                    
                    except Exception as e:
                        stats['total_failed'] += 1
                        error_msg = f"{model_name}_{model_id}_ws{window_size}: {str(e)}"
                        stats['errors'].append(error_msg)
                        logger.error(f"✗ Exception: {error_msg}")
        
        # Summary
        logger.info(f"\n{'='*80}")
        logger.info(f"EMBEDDING EXTRACTION - SUMMARY")
        logger.info(f"{'='*80}")
        logger.info(f"Total started: {stats['total_started']}")
        logger.info(f"Total extracted: {stats['total_succeeded']}")
        logger.info(f"Total skipped: {stats['total_skipped']}")
        logger.info(f"  - Existing embeddings: {stats['total_skipped_existing']}")
        logger.info(f"  - Unsupported model/window pairs: {stats['total_skipped_unsupported']}")
        logger.info(f"Total failed: {stats['total_failed']}")
        
        if stats['errors']:
            logger.info(f"\nErrors:")
            for error in stats['errors']:
                logger.error(f"  - {error}")
        
        elapsed = (datetime.now() - stats['start_time']).total_seconds() / 60
        logger.info(f"\nTotal time: {elapsed:.1f} minutes")
        logger.info(f"{'='*80}")
        
        return stats


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract embeddings from foundation models")
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--window-sizes', type=int, nargs='+', default=None, help='Window sizes to process')
    parser.add_argument('--models', type=str, nargs='+', default=None, help='Model families or model IDs to process')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = EmbeddingExtractor(device=args.device)
    
    # Optional filtering by model family / model id
    selected_models_config = MODELS_CONFIG
    if args.models:
        requested = {item.strip() for item in args.models if item and item.strip()}
        filtered = {}

        for model_name, config in MODELS_CONFIG.items():
            model_ids = list(config.get('model_ids', []))
            matched_ids = []

            for model_id in model_ids:
                model_short = model_id.split('/')[-1]
                if (
                    model_name in requested
                    or model_id in requested
                    or model_short in requested
                ):
                    matched_ids.append(model_id)

            if matched_ids:
                filtered_config = dict(config)
                filtered_config['model_ids'] = matched_ids
                filtered[model_name] = filtered_config

        if not filtered:
            available = []
            for family, cfg in MODELS_CONFIG.items():
                available.append(family)
                available.extend(cfg.get('model_ids', []))
            raise ValueError(
                "No models matched --models selection. "
                f"Requested={sorted(requested)}. "
                f"Available entries={available}"
            )

        selected_models_config = filtered

    # Extract embeddings
    stats = extractor.extract_all(
        window_sizes=args.window_sizes or WINDOW_SIZES,
        models_config=selected_models_config
    )
    
    # Print summary
    if stats['total_succeeded'] == stats['total_started']:
        print("\n✓ All embeddings extracted successfully!")
        return 0
    else:
        print(f"\n✗ {stats['total_failed']} extractions failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
