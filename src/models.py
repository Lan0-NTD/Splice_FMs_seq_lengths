"""
Foundation Model loader for DNA sequence analysis
Supports: HyenaDNA, DNABert, Nucleotide Transformer
"""

from typing import Dict, Any, Tuple, TYPE_CHECKING
import logging

# Import torch first to check if it's available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("torch not installed")

try:
    from transformers import AutoTokenizer, AutoModel, PreTrainedModel
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
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModel.from_pretrained(
                model_id,
                trust_remote_code=True
            )
            
            # Set pad token if not already set
            if tokenizer.pad_token is None:
                if tokenizer.eos_token:
                    tokenizer.pad_token = tokenizer.eos_token
                elif hasattr(tokenizer, 'unk_token') and tokenizer.unk_token:
                    tokenizer.pad_token = tokenizer.unk_token
                else:
                    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
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
