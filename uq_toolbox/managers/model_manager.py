# --- Standard Library ---
import gc
import getpass
import os
from typing import Any, Optional, Union

# --- Third-Party / ML Libraries ---
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)

# --- LM-Polygraph ---
from lm_polygraph.model_adapters import VisualWhiteboxModel
from lm_polygraph.utils.generation_parameters import GenerationParameters
from lm_polygraph.utils.model import BlackboxModel, WhiteboxModel

# --- LangChain ---
from langchain_openai import ChatOpenAI

# --- Local Imports ---
from .llama_cpp_manager import LlamaCppManager


class UQModelManager:
    """
    Manages LM-Polygraph and UQLM model instances.

    Each registered alias can independently operate in white-box or
    black-box mode.
    """

    def __init__(
        self,
        default_polygraph_mode: str = "black",
        default_uqlm_mode: str = "black",
    ) -> None:
        self.default_polygraph_mode = (
            default_polygraph_mode.strip().lower()
        )
        self.default_uqlm_mode = (
            default_uqlm_mode.strip().lower()
        )

        self.polygraph_models: dict[str, Any] = {}
        self.langchain_llms: dict[str, Any] = {}

        self.polygraph_modes: dict[str, str] = {}
        self.uqlm_modes: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def list_active_models(self) -> None:
        """Print the currently registered model aliases and modes."""

        print("\n" + "=" * 55)
        print("LIVE UNCERTAINTY QUANTIFICATION REGISTRIES")
        print("=" * 55)

        poly_aliases = list(self.polygraph_models)
        uqlm_aliases = list(self.langchain_llms)

        print(
            f"LM-Polygraph framework "
            f"({len(poly_aliases)} active):"
        )

        if not poly_aliases:
            print("  No active aliases.")

        for alias in poly_aliases:
            mode = self.polygraph_modes.get(
                alias,
                "unknown",
            ).upper()

            print(
                f"  Alias: {alias} — "
                f"Operational mode: {mode}-BOX"
            )

        print(
            f"\nUQLM framework "
            f"({len(uqlm_aliases)} active):"
        )

        if not uqlm_aliases:
            print("  No active aliases.")

        for alias in uqlm_aliases:
            mode = self.uqlm_modes.get(
                alias,
                "unknown",
            ).upper()

            print(
                f"  Alias: {alias} — "
                f"Operational mode: {mode}-BOX"
            )

        print("=" * 55 + "\n")

    # ------------------------------------------------------------------
    # LM-Polygraph model loading
    # ------------------------------------------------------------------

    def add_polygraph_model(
        self,
        model_id: str,
        alias: str = "default",
        mode: Optional[str] = None,
        is_multimodal: bool = False,
        quant_mode: str = "none",
        temperature: float = 0.7,
        max_new_tokens: int = 30,
        do_sample: bool = True,
        device_map: Union[str, dict] = "auto",
        attn_implementation: Optional[str] = "eager",
        **hf_extra_kwargs: Any,
    ) -> None:
        """
        Add an LM-Polygraph model to the registry.

        White-box models are loaded locally through Transformers.
        Black-box models use LM-Polygraph's API wrapper.
        """

        current_mode = (
            mode.strip().lower()
            if mode
            else self.default_polygraph_mode
        )

        if current_mode == "none":
            print(
                f"Polygraph mode is 'none'. "
                f"Skipping alias '{alias}'."
            )
            return

        if current_mode not in {"white", "black"}:
            raise ValueError(
                "Polygraph mode must be 'white', 'black', "
                "or 'none'."
            )

        quant_mode = quant_mode.strip().lower()

        if quant_mode not in {"none", "4bit", "8bit"}:
            raise ValueError(
                "quant_mode must be 'none', '4bit', or '8bit'."
            )

        print(
            f"\nAdding LM-Polygraph [{alias}] "
            f"in {current_mode.upper()}-BOX mode..."
        )

        self.polygraph_modes[alias] = current_mode

        generation_parameters = GenerationParameters(
            temperature=temperature,
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
        )

        if current_mode == "white":
            self._add_whitebox_polygraph_model(
                model_id=model_id,
                alias=alias,
                is_multimodal=is_multimodal,
                quant_mode=quant_mode,
                device_map=device_map,
                attn_implementation=attn_implementation,
                generation_parameters=generation_parameters,
                **hf_extra_kwargs,
            )
            return

        self._add_blackbox_polygraph_model(
            model_id=model_id,
            alias=alias,
            generation_parameters=generation_parameters,
        )

    def _add_whitebox_polygraph_model(
        self,
        model_id: str,
        alias: str,
        is_multimodal: bool,
        quant_mode: str,
        device_map: Union[str, dict],
        attn_implementation: Optional[str],
        generation_parameters: GenerationParameters,
        **hf_extra_kwargs: Any,
    ) -> None:
        """Load a local white-box model."""

        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            hf_token = getpass.getpass(
                "Paste your Hugging Face token: "
            )
            os.environ["HF_TOKEN"] = hf_token

        model_loader_class = hf_extra_kwargs.pop(
            "model_loader_class",
            AutoModelForImageTextToText,
        )

        model_kwargs: dict[str, Any] = {
            "device_map": (
                device_map
                if torch.cuda.is_available()
                else None
            ),
            "token": hf_token,
        }

        if attn_implementation:
            model_kwargs[
                "attn_implementation"
            ] = attn_implementation

        model_kwargs.setdefault(
            "trust_remote_code",
            True,
        )

        model_kwargs.update(hf_extra_kwargs)

        if quant_mode == "4bit":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit loading requires a CUDA-enabled GPU."
                )

            model_kwargs[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )

        elif quant_mode == "8bit":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "8-bit loading requires a CUDA-enabled GPU."
                )

            model_kwargs[
                "quantization_config"
            ] = BitsAndBytesConfig(
                load_in_8bit=True,
            )

        elif "torch_dtype" not in model_kwargs:
            model_kwargs["torch_dtype"] = (
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            )

        if is_multimodal:
            processor = AutoProcessor.from_pretrained(
                model_id,
                token=hf_token,
                trust_remote_code=model_kwargs.get(
                    "trust_remote_code",
                    True,
                ),
            )

            base_model = model_loader_class.from_pretrained(
                model_id,
                **model_kwargs,
            )

            self.polygraph_models[
                alias
            ] = VisualWhiteboxModel(
                model=base_model,
                processor_visual=processor,
                model_path=model_id,
                generation_parameters=generation_parameters,
            )

            return

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            token=hf_token,
            trust_remote_code=model_kwargs.get(
                "trust_remote_code",
                True,
            ),
        )

        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has neither a pad token nor "
                    "an EOS token."
                )

            tokenizer.pad_token = tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )

        base_model.eval()

        self.polygraph_models[alias] = WhiteboxModel(
            model=base_model,
            tokenizer=tokenizer,
            model_path=model_id,
            generation_parameters=generation_parameters,
        )

    def _add_blackbox_polygraph_model(
        self,
        model_id: str,
        alias: str,
        generation_parameters: GenerationParameters,
    ) -> None:
        """Register an API-based LM-Polygraph model."""

        openai_api_key = os.environ.get(
            "OPENAI_API_KEY"
        )

        if not openai_api_key:
            openai_api_key = getpass.getpass(
                "Paste your OpenAI API key: "
            )
            os.environ[
                "OPENAI_API_KEY"
            ] = openai_api_key

        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            hf_token = getpass.getpass(
                "Paste your Hugging Face token: "
            )
            os.environ["HF_TOKEN"] = hf_token

        self.polygraph_models[alias] = BlackboxModel(
            model_path=model_id,
            openai_api_key=openai_api_key,
            hf_api_token=hf_token,
            supports_logprobs=True,
            generation_parameters=generation_parameters,
        )

    # ------------------------------------------------------------------
    # UQLM model loading
    # ------------------------------------------------------------------

    def add_uqlm_model(
        self,
        provider: str,
        model_id: Optional[str] = None,
        alias: str = "default",
        mode: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        custom_llm: Optional[Any] = None,
        ollama_url: str = "http://localhost:11434",
        auto_install: bool = False,
        **uqlm_kwargs: Any,
    ) -> None:
        """Add a LangChain-compatible model for UQLM."""

        current_mode = (
            mode.strip().lower()
            if mode
            else self.default_uqlm_mode
        )

        if current_mode == "none":
            return

        if current_mode not in {"white", "black"}:
            raise ValueError(
                "UQLM mode must be 'white', 'black', "
                "or 'none'."
            )

        provider = provider.strip().lower()

        if (
            current_mode == "white"
            and provider in {"anthropic", "google"}
        ):
            print(
                f"Provider '{provider}' does not expose the "
                "required token log probabilities. "
                "Using black-box mode instead."
            )
            current_mode = "black"

        self.uqlm_modes[alias] = current_mode

        print(
            f"\nAdding UQLM [{alias}] "
            f"in {current_mode.upper()}-BOX mode..."
        )

        if provider == "custom":
            if custom_llm is None:
                raise ValueError(
                    "custom_llm must be provided when "
                    "provider='custom'."
                )

            self.langchain_llms[alias] = custom_llm
            return

        if not model_id:
            raise ValueError(
                "model_id must be provided for this provider."
            )

        if provider == "openai":
            self._add_openai_uqlm_model(
                alias=alias,
                model_id=model_id,
                current_mode=current_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                **uqlm_kwargs,
            )

        elif provider == "google":
            self._add_google_uqlm_model(
                alias=alias,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **uqlm_kwargs,
            )

        elif provider == "anthropic":
            self._add_anthropic_uqlm_model(
                alias=alias,
                model_id=model_id,
                temperature=temperature,
                max_tokens=max_tokens,
                **uqlm_kwargs,
            )

        elif provider == "llamacpp":
            self._add_llamacpp_uqlm_model(
                alias=alias,
                model_id=model_id,
                current_mode=current_mode,
                temperature=temperature,
                max_tokens=max_tokens,
                ollama_url=ollama_url,
                auto_install=auto_install,
                **uqlm_kwargs,
            )

        else:
            raise ValueError(
                "Unsupported provider. Use 'openai', "
                "'google', 'anthropic', 'llamacpp', "
                "or 'custom'."
            )

    def _add_openai_uqlm_model(
        self,
        alias: str,
        model_id: str,
        current_mode: str,
        temperature: float,
        max_tokens: Optional[int],
        **uqlm_kwargs: Any,
    ) -> None:
        """Register an OpenAI LangChain model."""

        if "OPENAI_API_KEY" not in os.environ:
            os.environ[
                "OPENAI_API_KEY"
            ] = getpass.getpass(
                "Paste your OpenAI API key: "
            )

        model_kwargs = (
            {"logprobs": True}
            if current_mode == "white"
            else {}
        )

        self.langchain_llms[alias] = ChatOpenAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            **uqlm_kwargs,
        )

    def _add_google_uqlm_model(
        self,
        alias: str,
        model_id: str,
        temperature: float,
        max_tokens: Optional[int],
        **uqlm_kwargs: Any,
    ) -> None:
        """Register a Google LangChain model."""

        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )

        if "GOOGLE_API_KEY" not in os.environ:
            os.environ[
                "GOOGLE_API_KEY"
            ] = getpass.getpass(
                "Paste your Google API key: "
            )

        self.langchain_llms[
            alias
        ] = ChatGoogleGenerativeAI(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **uqlm_kwargs,
        )

    def _add_anthropic_uqlm_model(
        self,
        alias: str,
        model_id: str,
        temperature: float,
        max_tokens: Optional[int],
        **uqlm_kwargs: Any,
    ) -> None:
        """Register an Anthropic LangChain model."""

        from langchain_anthropic import ChatAnthropic

        if "ANTHROPIC_API_KEY" not in os.environ:
            os.environ[
                "ANTHROPIC_API_KEY"
            ] = getpass.getpass(
                "Paste your Anthropic API key: "
            )

        self.langchain_llms[alias] = ChatAnthropic(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **uqlm_kwargs,
        )

    def _add_llamacpp_uqlm_model(
        self,
        alias: str,
        model_id: str,
        current_mode: str,
        temperature: float,
        max_tokens: Optional[int],
        ollama_url: str,
        auto_install: bool,
        **uqlm_kwargs: Any,
    ) -> None:
        """Register a local Llama.cpp-compatible model."""

        print(
            f"\nInitialising LlamaCpp for [{alias}]..."
        )

        manager = LlamaCppManager(
            base_url=ollama_url,
        )

        if not manager.load_model(
            model_name=model_id,
        ):
            raise RuntimeError(
                f"Failed to initialise LlamaCpp model "
                f"for alias '{alias}'."
            )

        model_kwargs = (
            {"logprobs": True}
            if current_mode == "white"
            else {}
        )

        self.langchain_llms[alias] = ChatOpenAI(
            base_url=f"{manager.base_url}/v1",
            api_key="llama-cpp-local",
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs=model_kwargs,
            **uqlm_kwargs,
        )

        print(
            f"LlamaCpp instance active at "
            f"{manager.base_url}/v1"
        )

    # ------------------------------------------------------------------
    # Model cleanup
    # ------------------------------------------------------------------

    def remove_models(
        self,
        aliases: Optional[
            Union[list[str], str]
        ] = None,
    ) -> None:
        """Remove selected models, or all models, from memory."""

        if aliases is None:
            targets = list(
                set(
                    list(self.polygraph_models)
                    + list(self.langchain_llms)
                )
            )

        elif isinstance(aliases, str):
            targets = [aliases.strip()]

        else:
            targets = [
                alias.strip()
                for alias in aliases
            ]

        if not targets:
            print(
                "No active models were found to remove."
            )
            return

        print(
            f"\nRemoving model aliases: {targets}"
        )

        for alias in targets:
            removed = False

            if alias in self.polygraph_models:
                wrapper = self.polygraph_models.pop(
                    alias
                )

                if isinstance(
                    wrapper,
                    VisualWhiteboxModel,
                ):
                    if hasattr(wrapper, "model"):
                        wrapper.model = None

                    if hasattr(
                        wrapper,
                        "processor_visual",
                    ):
                        wrapper.processor_visual = None

                elif isinstance(
                    wrapper,
                    WhiteboxModel,
                ):
                    if hasattr(wrapper, "model"):
                        wrapper.model = None

                    if hasattr(wrapper, "tokenizer"):
                        wrapper.tokenizer = None

                self.polygraph_modes.pop(
                    alias,
                    None,
                )

                removed = True

            if alias in self.langchain_llms:
                self.langchain_llms.pop(
                    alias,
                    None,
                )

                self.uqlm_modes.pop(
                    alias,
                    None,
                )

                removed = True

            if removed:
                print(
                    f"  Removed alias: {alias}"
                )
            else:
                print(
                    f"  Alias not found: {alias}"
                )

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

            print("CUDA cache cleared.")
        else:
            print("Garbage collection completed.")


def initialize_uq_models(
    polygraph_models: Optional[
        Union[str, dict]
    ] = None,
    polygraph_is_multimodal: bool = False,
    uqlm_models: Optional[dict] = None,
    p_mode: Optional[str] = None,
    u_mode: Optional[str] = None,
    quant_choice: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 256,
    ollama_url: str = "http://localhost:11434",
    auto_install: bool = False,
    polygraph_kwargs: Optional[dict] = None,
    uqlm_kwargs: Optional[dict] = None,
) -> UQModelManager:
    """
    Initialise and register multiple LM-Polygraph and UQLM models.
    """

    print("\n" + "=" * 60)
    print("INITIALISING UQ MODEL ENVIRONMENT")
    print("=" * 60)

    polygraph_kwargs = (
        polygraph_kwargs or {}
    )
    uqlm_kwargs = uqlm_kwargs or {}

    if isinstance(polygraph_models, str):
        polygraph_config = {
            "default": {
                "model_id": polygraph_models,
            }
        }

    elif isinstance(polygraph_models, dict):
        polygraph_config = {
            alias: (
                {"model_id": value}
                if isinstance(value, str)
                else value.copy()
            )
            for alias, value
            in polygraph_models.items()
        }

    else:
        polygraph_config = {}

    uqlm_config = uqlm_models or {}

    resolved_p_mode = p_mode or "black"
    resolved_u_mode = u_mode or "black"

    manager = UQModelManager(
        default_polygraph_mode=resolved_p_mode,
        default_uqlm_mode=resolved_u_mode,
    )

    for alias, config in polygraph_config.items():
        if "model_id" not in config:
            raise ValueError(
                f"Missing model_id for Polygraph "
                f"alias '{alias}'."
            )

        nested_kwargs = config.get(
            "kwargs",
            {},
        ).copy()

        reserved_keys = {
            "model_id",
            "mode",
            "quant_mode",
            "temperature",
            "max_new_tokens",
            "do_sample",
            "is_multimodal",
            "kwargs",
        }

        flat_kwargs = {
            key: value
            for key, value in config.items()
            if key not in reserved_keys
        }

        merged_kwargs = {
            **polygraph_kwargs,
            **flat_kwargs,
            **nested_kwargs,
        }

        manager.add_polygraph_model(
            model_id=config["model_id"],
            alias=alias,
            mode=config.get(
                "mode",
                resolved_p_mode,
            ),
            quant_mode=config.get(
                "quant_mode",
                quant_choice or "none",
            ),
            temperature=config.get(
                "temperature",
                temperature,
            ),
            max_new_tokens=config.get(
                "max_new_tokens",
                max_tokens,
            ),
            do_sample=config.get(
                "do_sample",
                True,
            ),
            is_multimodal=config.get(
                "is_multimodal",
                polygraph_is_multimodal,
            ),
            **merged_kwargs,
        )

    for alias, config in uqlm_config.items():
        if "provider" not in config:
            raise ValueError(
                f"Missing provider for UQLM "
                f"alias '{alias}'."
            )

        nested_kwargs = config.get(
            "kwargs",
            {},
        ).copy()

        reserved_keys = {
            "provider",
            "model_id",
            "mode",
            "temperature",
            "max_tokens",
            "custom_llm",
            "ollama_url",
            "auto_install",
            "kwargs",
        }

        flat_kwargs = {
            key: value
            for key, value in config.items()
            if key not in reserved_keys
        }

        merged_kwargs = {
            **uqlm_kwargs,
            **flat_kwargs,
            **nested_kwargs,
        }

        manager.add_uqlm_model(
            provider=config["provider"],
            model_id=config.get("model_id"),
            alias=alias,
            mode=config.get(
                "mode",
                resolved_u_mode,
            ),
            temperature=config.get(
                "temperature",
                temperature,
            ),
            max_tokens=config.get(
                "max_tokens",
                max_tokens,
            ),
            custom_llm=config.get(
                "custom_llm",
            ),
            ollama_url=config.get(
                "ollama_url",
                ollama_url,
            ),
            auto_install=config.get(
                "auto_install",
                auto_install,
            ),
            **merged_kwargs,
        )

    print("\n" + "=" * 60)
    print("UQ MODEL ENVIRONMENT READY")
    print("=" * 60)

    return manager
