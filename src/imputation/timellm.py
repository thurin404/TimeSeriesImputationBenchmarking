import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import TimeLLM
except ImportError:
    print("TimeLLM not available in current PyPOTS version")
    TimeLLM = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class TimeLLMImputer(BasePypotsImputer):
    """Time-LLM imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("timellm")
        # Force low memory mode for Time-LLM due to LLM memory requirements
        self._env_overrides["low_memory_mode"] = True
    
    def get_model_class(self) -> Type:
        """Return the TimeLLM model class"""
        if TimeLLM is None:
            raise ImportError("TimeLLM not available in current PyPOTS version")
        return TimeLLM
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return Time-LLM-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "llm_model_type": params["llm_model_type"],
            "patch_size": params["patch_size"],
            "patch_stride": params["patch_stride"],
            "d_llm": params["d_llm"],
            "d_model": params["d_model"],
            "d_ffn": params["d_ffn"],
            "n_heads": params["n_heads"],
            "dropout": params["dropout"],
            "domain_prompt_content": params["domain_prompt_content"],
            "ORT_weight": params.get("ORT_weight", 1.0),
            "MIT_weight": params.get("MIT_weight", 1.0),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best",
            "device": "cpu"  # Force CPU for Time-LLM due to GPU memory constraints
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define Time-LLM hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Very conservative parameters for low memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 3),
                "llm_model_type": trial.suggest_categorical("llm_model_type", ["GPT2"]),
                "patch_size": trial.suggest_categorical("patch_size", [8, 16]),
                "patch_stride": trial.suggest_categorical("patch_stride", [4, 8]),
                "d_llm": trial.suggest_categorical("d_llm", [768]),  # Standard GPT2 dimension
                "d_model": trial.suggest_categorical("d_model", [64, 128]),
                "d_ffn": trial.suggest_categorical("d_ffn", [256]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.1),
                "domain_prompt_content": "The dataset consists of multivariate time series with missing values that need to be imputed.",
                "ORT_weight": 1.0,
                "MIT_weight": 1.0,
                "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                "epochs": trial.suggest_int("epochs", 25, 50, step=25),
                "patience": trial.suggest_int("patience", 5, 10, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "llm_model_type": trial.suggest_categorical("llm_model_type", ["GPT2", "BERT"]),
                "patch_size": trial.suggest_categorical("patch_size", [8, 16, 24, 32]),
                "patch_stride": trial.suggest_categorical("patch_stride", [4, 8, 12, 16]),
                "d_llm": trial.suggest_categorical("d_llm", [768, 1024]),
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256]),
                "d_ffn": trial.suggest_categorical("d_ffn", [256, 512, 1024]),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8, 16]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "domain_prompt_content": "The dataset consists of multivariate time series with missing values that need to be imputed.",
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 2.0, step=0.5),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 2.0, step=0.5),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use TimeLLMImputer class instead"""
    imputer = TimeLLMImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use TimeLLMImputer class instead"""
    imputer = TimeLLMImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/timellm.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run Time-LLM imputer
    imputer = TimeLLMImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()
