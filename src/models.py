"""
Foundation Model loader for DNA sequence analysis
Supports: HyenaDNA, DNABert, Nucleotide Transformer
"""

from typing import Dict, Any, Tuple, TYPE_CHECKING
import logging
import sys
from pathlib import Path

# Import torch first to check if it's available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch not installed")

try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM, AutoConfig, PreTrainedModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not installed")
    # Define dummy types if transformers not available
    if TYPE_CHECKING:
        PreTrainedModel = object
    else:
        PreTrainedModel = object

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class FoundationModelLoader:
    """Loader for various DNA foundation models"""
    
    def __init__(self, device: str = None):
        # Set device with proper fallback
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.models_cache = {}
        self.tokenizers_cache = {}
    
    def load_hyena_dna(self, model_id: str = "hyenadna-tiny-1d-d256") -> Tuple[PreTrainedModel, Any]:
        """
        Load HyenaDNA model
        
        Args:
            model_id: Model identifier from HyenaDNA family
                     Options: hyenadna-tiny-1d-d256, hyenadna-small-1d-d256, hyenadna-medium0-1d-d256
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Please install it with: pip install transformers"
            )
        
        if model_id in self.models_cache:
            return self.models_cache[model_id], self.tokenizers_cache[model_id]
        
        try:
            # HyenaDNA models from huggingface
            model_name = f"facebook/{model_id}"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            self.models_cache[model_id] = model
            self.tokenizers_cache[model_id] = tokenizer
            
            logger.info(f"Successfully loaded HyenaDNA model: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load HyenaDNA model {model_id}: {str(e)}")
            raise
    
    def load_dna_bert(self, model_id: str = "zhihan1996/DNA_bert_5") -> Tuple[PreTrainedModel, Any]:
        """
        Load DNABert model
        
        Args:
            model_id: Model identifier from DNABert family
                     Options: zhihan1996/DNA_bert_3, zhihan1996/DNA_bert_4, zhihan1996/DNA_bert_5
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Please install it with: pip install transformers"
            )
        
        if model_id in self.models_cache:
            return self.models_cache[model_id], self.tokenizers_cache[model_id]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            self.models_cache[model_id] = model
            self.tokenizers_cache[model_id] = tokenizer
            
            logger.info(f"Successfully loaded DNABert model: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load DNABert model {model_id}: {str(e)}")
            raise
    
    def load_nucleotide_transformer(self, model_id: str = "lawerenz/nt_transformer_v2_500m_1000g") -> Tuple[PreTrainedModel, Any]:
        """
        Load Nucleotide Transformer model
        
        Args:
            model_id: Model identifier from Nucleotide Transformer family
                     Options: lawerenz/nt_transformer_v2_100m_1000g, 
                             lawerenz/nt_transformer_v2_250m_1000g,
                             lawerenz/nt_transformer_v2_500m_1000g
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Please install it with: pip install transformers"
            )
        
        if model_id in self.models_cache:
            return self.models_cache[model_id], self.tokenizers_cache[model_id]
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            self.models_cache[model_id] = model
            self.tokenizers_cache[model_id] = tokenizer
            
            logger.info(f"Successfully loaded Nucleotide Transformer model: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load Nucleotide Transformer model {model_id}: {str(e)}")
            raise
    
    def load_model_by_name(self, model_name: str, model_id: str) -> Tuple[PreTrainedModel, Any]:
        """
        Load a model by name and identifier
        
        Args:
            model_name: One of ["HyenaDNA", "DNABert", "NucleotideTransformer"]
            model_id: Specific model identifier
        
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name == "HyenaDNA":
            # HyenaDNA models - just load directly since we're using alternatives
            return self._load_model_by_id(model_id, model_name)
        elif model_name == "DNABert":
            return self._load_model_by_id(model_id, model_name)
        elif model_name == "NucleotideTransformer":
            return self._load_model_by_id(model_id, model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def _load_model_by_id(self, model_id: str, model_name: str = "") -> Tuple[PreTrainedModel, Any]:
        """Load any model by direct model ID"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library not installed. "
                "Please install it with: pip install transformers"
            )
        
        if model_id in self.models_cache:
            return self.models_cache[model_id], self.tokenizers_cache[model_id]
        
        try:
            config = None
            is_dnabert = (
                model_name == "DNABert"
                or "dnabert" in model_id.lower()
            )

            # Prefer standard Transformers implementation for DNABERT, but auto-fallback
            # to trust_remote_code=True when the repo requires custom code.
            tokenizer_trust_remote_code = False if is_dnabert else True
            model_trust_remote_code = False if is_dnabert else True

            def _requires_remote_code(error: Exception) -> bool:
                error_text = str(error).lower()
                return (
                    "trust_remote_code=true" in error_text
                    or "contains custom code" in error_text
                    or "must be executed to correctly load" in error_text
                )

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id,
                    trust_remote_code=tokenizer_trust_remote_code
                )
            except Exception as e:
                if is_dnabert and _requires_remote_code(e):
                    logger.warning(f"{model_id} requires custom code; retrying tokenizer load with trust_remote_code=True")
                    tokenizer_trust_remote_code = True
                    model_trust_remote_code = True
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_id,
                        trust_remote_code=True
                    )
                else:
                    raise

            def _apply_safe_attention_settings(cfg):
                attention_flags = [
                    "use_flash_attn",
                    "flash_attn",
                    "use_flash_attention",
                    "flash_attention",
                    "flash_attn_triton",
                    "flash_attention_2",
                    "use_flash_attention_2",
                    "_flash_attn_2_enabled",
                    "use_triton",
                    "fused_attention",
                    "use_xformers",
                    "xformers",
                    "xformers_attention",
                    "memory_efficient_attention",
                ]
                for attr_name in attention_flags:
                    if hasattr(cfg, attr_name):
                        setattr(cfg, attr_name, False)

                if hasattr(cfg, "attn_implementation"):
                    setattr(cfg, "attn_implementation", "eager")

            def _disable_dnabert_remote_flash_attention(loaded_model):
                """Force DNABERT remote module to skip Triton flash attention kernels."""
                if not is_dnabert:
                    return

                patched_modules = 0

                # 1) Patch the model's own Python module if present
                try:
                    module_name = loaded_model.__class__.__module__
                    module_obj = sys.modules.get(module_name)
                    if module_obj is not None and hasattr(module_obj, "flash_attn_qkvpacked_func"):
                        setattr(module_obj, "flash_attn_qkvpacked_func", None)
                        patched_modules += 1
                except Exception as patch_err:
                    logger.warning(f"Could not patch DNABERT model module flash attention: {patch_err}")

                # 2) Patch all imported DNABERT bert_layers modules (covers nested/aliased imports)
                for module_name, module_obj in list(sys.modules.items()):
                    if module_obj is None:
                        continue
                    if "dnabert" in module_name.lower() and "bert_layers" in module_name.lower():
                        if hasattr(module_obj, "flash_attn_qkvpacked_func"):
                            try:
                                setattr(module_obj, "flash_attn_qkvpacked_func", None)
                                patched_modules += 1
                            except Exception:
                                pass

                if patched_modules > 0:
                    logger.info(f"Patched DNABERT remote flash attention in {patched_modules} module(s); using PyTorch attention path")

            def _ensure_config_compat(cfg):
                defaults = {
                    "is_decoder": False,
                    "add_cross_attention": False,
                    "cross_attention_hidden_size": None,
                    "pad_token_id": tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
                    "bos_token_id": getattr(tokenizer, "bos_token_id", None),
                    "eos_token_id": getattr(tokenizer, "eos_token_id", None),
                    "sep_token_id": getattr(tokenizer, "sep_token_id", None),
                    "cls_token_id": getattr(tokenizer, "cls_token_id", None),
                }
                for attr_name, attr_value in defaults.items():
                    if not hasattr(cfg, attr_name):
                        setattr(cfg, attr_name, attr_value)

            try:
                config = AutoConfig.from_pretrained(model_id, trust_remote_code=model_trust_remote_code)
                _ensure_config_compat(config)
                _apply_safe_attention_settings(config)
            except Exception as e:
                if is_dnabert and _requires_remote_code(e):
                    logger.warning(f"{model_id} requires custom code; retrying config load with trust_remote_code=True")
                    model_trust_remote_code = True
                    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    _ensure_config_compat(config)
                    _apply_safe_attention_settings(config)
                else:
                    logger.warning(f"Could not pre-load config for {model_id}: {e}")

            model_load_kwargs = {
                "trust_remote_code": model_trust_remote_code,
                "low_cpu_mem_usage": False,
            }
            if config is not None:
                model_load_kwargs["config"] = config
            model_load_kwargs["attn_implementation"] = "eager"

            try:
                model = AutoModel.from_pretrained(model_id, **model_load_kwargs)
            except AttributeError as e:
                error_text = str(e)
                if "has no attribute" in error_text or "is_decoder" in error_text or "add_cross_attention" in error_text:
                    logger.warning(f"Retrying {model_id} after patching config compatibility attrs")
                    if config is None:
                        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
                    _ensure_config_compat(config)
                    model = AutoModel.from_pretrained(
                        model_id,
                        trust_remote_code=True,
                        config=config
                    )
                else:
                    raise
            except Exception as e:
                error_text = str(e)
                if is_dnabert and _requires_remote_code(e):
                    logger.warning(f"{model_id} requires custom code; retrying model load with trust_remote_code=True")
                    retry_kwargs = dict(model_load_kwargs)
                    retry_kwargs["trust_remote_code"] = True
                    model = AutoModel.from_pretrained(model_id, **retry_kwargs)
                else:
                    mismatch_markers = [
                        "ignore_mismatched_sizes",
                        "size mismatch",
                        "MISMATCH",
                    ]
                    if any(marker in error_text for marker in mismatch_markers):
                        logger.warning(
                            f"Retrying {model_id} with ignore_mismatched_sizes=True due to checkpoint/model shape mismatch"
                        )
                        retry_kwargs = dict(model_load_kwargs)
                        retry_kwargs["ignore_mismatched_sizes"] = True
                        model = AutoModel.from_pretrained(model_id, **retry_kwargs)
                    elif "device meta" in error_text.lower() or "tensor on device meta" in error_text.lower():
                        logger.warning(
                            f"Retrying {model_id} with low_cpu_mem_usage=False and ignore_mismatched_sizes=True to resolve meta-device tensors"
                        )
                        retry_kwargs = dict(model_load_kwargs)
                        retry_kwargs["low_cpu_mem_usage"] = False
                        retry_kwargs["ignore_mismatched_sizes"] = True
                        try:
                            model = AutoModel.from_pretrained(model_id, **retry_kwargs)
                        except Exception as inner_e:
                            inner_error_text = str(inner_e).lower()
                            if is_dnabert and ("device meta" in inner_error_text or "tensor on device meta" in inner_error_text):
                                logger.warning(
                                    f"Falling back to DNABERT MLM loader for {model_id} and using its backbone encoder"
                                )
                                try:
                                    mlm_model = AutoModelForMaskedLM.from_pretrained(
                                        model_id,
                                        trust_remote_code=True,
                                        config=config,
                                        low_cpu_mem_usage=False,
                                        ignore_mismatched_sizes=True,
                                    )
                                    if hasattr(mlm_model, "bert"):
                                        model = mlm_model.bert
                                    elif hasattr(mlm_model, "base_model"):
                                        model = mlm_model.base_model
                                    else:
                                        model = mlm_model
                                except Exception as final_e:
                                    final_error_text = str(final_e).lower()
                                    if "device meta" in final_error_text or "tensor on device meta" in final_error_text:
                                        logger.warning(
                                            f"Final fallback: manual state_dict load for {model_id} to bypass meta tensors"
                                        )
                                        model = AutoModel.from_config(config, trust_remote_code=True)

                                        if not HF_HUB_AVAILABLE:
                                            raise RuntimeError(
                                                "huggingface_hub is required for manual DNABERT weight loading"
                                            )

                                        weight_file = hf_hub_download(
                                            repo_id=model_id,
                                            filename="pytorch_model.bin"
                                        )
                                        state_dict = torch.load(Path(weight_file), map_location="cpu")
                                        if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                                            state_dict = state_dict["state_dict"]

                                        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                                        if missing_keys:
                                            logger.warning(f"Manual load missing keys: {len(missing_keys)}")
                                        if unexpected_keys:
                                            logger.warning(f"Manual load unexpected keys: {len(unexpected_keys)}")
                                    else:
                                        raise
                            else:
                                raise
                    else:
                        raise

            if hasattr(model, "config") and model.config is not None:
                _apply_safe_attention_settings(model.config)

            _disable_dnabert_remote_flash_attention(model)
            
            # Set pad token if not already set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    model.resize_token_embeddings(len(tokenizer))
            
            # Move to device
            model = model.to(self.device)
            model.eval()
            self.models_cache[model_id] = model
            self.tokenizers_cache[model_id] = tokenizer
            
            logger.info(f"Successfully loaded {model_name} model: {model_id}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load {model_name} model {model_id}: {str(e)}")
            raise
    
    def load_all_models(self, models_config: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Tuple]]:
        """
        Load all models from config
        
        Args:
            models_config: Configuration dictionary with model specifications
        
        Returns:
            Dictionary of all loaded models and tokenizers
        """
        all_models = {}
        
        for model_name, config in models_config.items():
            all_models[model_name] = {}
            
            if model_name == "HyenaDNA":
                for model_id in config["model_ids"]:
                    try:
                        model, tokenizer = self.load_hyena_dna(model_id)
                        all_models[model_name][model_id] = {"model": model, "tokenizer": tokenizer}
                    except Exception as e:
                        logger.error(f"Error loading {model_name} {model_id}: {str(e)}")
            
            elif model_name == "DNABert":
                for model_id in config["model_ids"]:
                    try:
                        model, tokenizer = self.load_dna_bert(model_id)
                        all_models[model_name][model_id] = {"model": model, "tokenizer": tokenizer}
                    except Exception as e:
                        logger.error(f"Error loading {model_name} {model_id}: {str(e)}")
            
            elif model_name == "NucleotideTransformer":
                for model_id in config["model_ids"]:
                    try:
                        model, tokenizer = self.load_nucleotide_transformer(model_id)
                        all_models[model_name][model_id] = {"model": model, "tokenizer": tokenizer}
                    except Exception as e:
                        logger.error(f"Error loading {model_name} {model_id}: {str(e)}")
        
        return all_models
    
    def get_model_info(self, model_name: str, model_id: str) -> Dict[str, Any]:
        """Get information about a loaded model"""
        if model_id not in self.models_cache:
            raise ValueError(f"Model {model_id} not loaded")
        
        model = self.models_cache[model_id]
        return {
            "model_name": model_name,
            "model_id": model_id,
            "num_parameters": sum(p.numel() for p in model.parameters()),
            "model_type": model.__class__.__name__,
            "hidden_size": getattr(model.config, "hidden_size", None),
            "num_hidden_layers": getattr(model.config, "num_hidden_layers", None),
        }
    
    def clear_cache(self):
        """Clear model and tokenizer cache"""
        self.models_cache.clear()
        self.tokenizers_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Model cache cleared")
