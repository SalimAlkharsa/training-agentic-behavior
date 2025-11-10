"""
Wrapper class for Qwen2.5-Coder-3B-Instruct model with quantization support.
"""

from typing import Optional, Dict, Any, List, Union
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from pathlib import Path
import warnings

from ..utils.metrics import MetricsRecorder


class Qwen2Wrapper:
    """
    Wrapper class for Qwen2.5-Coder-3B-Instruct model with support for
    quantization, device management, and integrated metrics tracking.
    """

    # Class-level cache for models (singleton pattern)
    _model_cache: Dict[str, Any] = {}
    _tokenizer_cache: Dict[str, Any] = {}

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-3B-Instruct",
        quantization: Optional[str] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        enable_metrics: bool = True,
        metrics_save_dir: str = "results/metrics"
    ):
        """
        Initialize the Qwen2 model wrapper.

        Args:
            model_name: HuggingFace model identifier
            quantization: Quantization mode - "4bit", "8bit", or None for full precision
            device: Device to use ("cuda", "cpu", or None for auto-detection)
            cache_dir: Directory to cache downloaded models
            use_cache: Whether to use cached model instances
            enable_metrics: Whether to enable metrics recording
            metrics_save_dir: Directory to save metrics
        """
        self.model_name = model_name
        self.quantization = quantization
        self.cache_dir = cache_dir
        self.use_cache = use_cache

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Validate device
        if self.device == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to MPS or CPU")
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        elif self.device == "mps" and not torch.backends.mps.is_available():
            warnings.warn("MPS not available, falling back to CPU")
            self.device = "cpu"

        # Initialize metrics recorder
        self.metrics_recorder = MetricsRecorder(save_dir=metrics_save_dir) if enable_metrics else None

        # Model and tokenizer will be loaded lazily
        self.model = None
        self.tokenizer = None
        self._cache_key = f"{model_name}_{quantization}_{self.device}"

    def load_model(self, force_reload: bool = False) -> None:
        """
        Load the model and tokenizer.

        Args:
            force_reload: Force reload even if cached version exists
        """
        # Check cache first
        if self.use_cache and not force_reload and self._cache_key in self._model_cache:
            self.model = self._model_cache[self._cache_key]
            self.tokenizer = self._tokenizer_cache[self._cache_key]
            print(f"Loaded model from cache: {self.model_name} ({self.quantization or 'full precision'})")
            return

        print(f"Loading model: {self.model_name}")
        print(f"Quantization: {self.quantization or 'None (full precision)'}")
        print(f"Device: {self.device}")

        # Configure quantization if specified
        quantization_config = None
        if self.quantization == "4bit":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif self.quantization == "8bit":
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True
            )

        # Load tokenizer with left-padding for decoder-only models
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
            padding_side='left'
        )

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_name,
            "cache_dir": self.cache_dir,
            "trust_remote_code": True,
        }

        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
            model_kwargs["device_map"] = "auto"
        else:
            # Use float16 for CUDA/MPS, float32 for CPU
            model_kwargs["dtype"] = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)

        # Move to device if not using device_map
        if quantization_config is None:
            self.model = self.model.to(self.device)

        # Cache the model and tokenizer
        if self.use_cache:
            self._model_cache[self._cache_key] = self.model
            self._tokenizer_cache[self._cache_key] = self.tokenizer

        print(f"Model loaded successfully on {self.device}")

    def generate(
        self,
        prompt: Union[str, List[str]],
        max_length: int = 512,
        top_k: int = 50,
        num_return_sequences: int = 1,
        do_sample: bool = True,
        record_metrics: bool = True,
        **kwargs
    ) -> Union[str, List[str]]:
        """
        Generate text from prompt(s).

        Args:
            prompt: Input prompt or list of prompts
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to generate per prompt
            do_sample: Whether to use sampling or greedy decoding
            record_metrics: Whether to record metrics for this generation
            **kwargs: Additional arguments for generation

        Returns:
            Generated text or list of generated texts
        """
        # Ensure model is loaded
        if self.model is None or self.tokenizer is None:
            self.load_model()

        # Handle single prompt vs batch
        is_batch = isinstance(prompt, list)
        prompts = prompt if is_batch else [prompt]

        # Start metrics recording if enabled
        if record_metrics and self.metrics_recorder:
            self.metrics_recorder.start_recording(
                operation_name="generate",
                model_name=self.model_name,
                quantization=self.quantization or "full",
                device=self.device,
                num_prompts=len(prompts),
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

        # Tokenize inputs
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)

        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sequences,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **kwargs
            )

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        # Calculate tokens generated
        output_tokens = outputs.shape[1] - input_length

        # Stop metrics recording if enabled
        if record_metrics and self.metrics_recorder:
            self.metrics_recorder.stop_recording(
                input_tokens=input_length,
                output_tokens=output_tokens,
                total_tokens=outputs.shape[1]
            )

        # Return single string if input was single string
        if not is_batch and num_return_sequences == 1:
            return generated_texts[0]

        return generated_texts

    def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **generation_kwargs
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            batch_size: Number of prompts to process at once
            **generation_kwargs: Arguments passed to generate()

        Returns:
            List of generated texts
        """
        all_outputs = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = self.generate(batch, **generation_kwargs)
            if isinstance(outputs, str):
                outputs = [outputs]
            all_outputs.extend(outputs)

        return all_outputs

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current memory usage.

        Returns:
            Dictionary with memory usage information
        """
        if self.metrics_recorder:
            return self.metrics_recorder.get_current_memory()
        return {}

    def save_metrics(self, filename: Optional[str] = None) -> Optional[Path]:
        """
        Save recorded metrics to file.

        Args:
            filename: Optional filename for metrics file

        Returns:
            Path to saved file, or None if metrics not enabled
        """
        if self.metrics_recorder:
            return self.metrics_recorder.save_metrics(filename)
        return None

    def print_metrics_summary(self) -> None:
        """Print summary of recorded metrics."""
        if self.metrics_recorder:
            self.metrics_recorder.print_summary()
        else:
            print("Metrics recording not enabled.")

    def clear_cache(self) -> None:
        """Clear the model cache."""
        if self._cache_key in self._model_cache:
            del self._model_cache[self._cache_key]
            del self._tokenizer_cache[self._cache_key]

        # Also clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

    @classmethod
    def clear_all_cache(cls) -> None:
        """Clear all cached models."""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return (
            f"Qwen2Wrapper(model={self.model_name}, "
            f"quantization={self.quantization or 'full'}, "
            f"device={self.device})"
        )
