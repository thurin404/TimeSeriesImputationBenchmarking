import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import GPT4TS
except ImportError:
    print("GPT4TS not available in current PyPOTS version")
    GPT4TS = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class GPT4TSImputer(BasePypotsImputer):
    """GPT4TS imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("gpt4ts")
        # Force low memory mode for GPT4TS due to GPU memory constraints
        self._env_overrides["low_memory_mode"] = True
    
    def get_model_class(self) -> Type:
        """Return the GPT4TS model class"""
        if GPT4TS is None:
            raise ImportError("GPT4TS not available in current PyPOTS version")
        return GPT4TS
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return GPT4TS-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "patch_size": params["patch_size"],
            "patch_stride": params["patch_stride"],
            "n_layers": params["n_layers"],
            "train_gpt_mlp": params.get("train_gpt_mlp", False),  # Default to False for stability
            "d_ffn": params["d_ffn"],
            "dropout": params["dropout"],
            "embed": params.get("embed", "fixed"),  # Default to fixed for stability
            "freq": params.get("freq", "h"),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define GPT4TS hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Very conservative parameters for low memory environments (3.68 GiB GPU)
            return {
                "patch_size": trial.suggest_categorical("patch_size", [8, 16]),
                "patch_stride": trial.suggest_categorical("patch_stride", [4, 8]),
                "n_layers": trial.suggest_int("n_layers", 2, 3),
                "train_gpt_mlp": trial.suggest_categorical("train_gpt_mlp", [False]),  # MLP training uses more memory
                "d_ffn": trial.suggest_categorical("d_ffn", [256]),  # Keep minimal
                "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.1),
                "embed": trial.suggest_categorical("embed", ["fixed"]),  # Fixed embeddings use less memory
                "freq": "h",  # Keep fixed for most use cases
                "batch_size": trial.suggest_categorical("batch_size", [8, 16]),  # Very small batches
                "epochs": trial.suggest_int("epochs", 25, 50, step=25),
                "patience": trial.suggest_int("patience", 5, 10, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "patch_size": trial.suggest_categorical("patch_size", [8, 16, 24, 32]),
                "patch_stride": trial.suggest_categorical("patch_stride", [4, 8, 12, 16]),
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "train_gpt_mlp": trial.suggest_categorical("train_gpt_mlp", [True, False]),
                "d_ffn": trial.suggest_categorical("d_ffn", [256, 512, 1024]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "embed": trial.suggest_categorical("embed", ["fixed", "learned"]),
                "freq": "h",  # Keep fixed for most use cases
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use GPT4TSImputer class instead"""
    imputer = GPT4TSImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use GPT4TSImputer class instead"""
    imputer = GPT4TSImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/gpt4ts.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run GPT4TS imputer
    imputer = GPT4TSImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()