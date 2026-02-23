import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import TimeMixerPP
except ImportError:
    print("TimeMixer not available in current PyPOTS version")
    TimeMixerPP = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class TimeMixerImputer(BasePypotsImputer):
    """TimeMixer imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("timemixer")
    
    def get_model_class(self) -> Type:
        """Return the TimeMixer model class"""
        if TimeMixerPP is None:
            raise ImportError("TimeMixer not available in current PyPOTS version")
        return TimeMixerPP

    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return TimeMixer-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_layers": params["n_layers"],
            "d_model": params["d_model"],
            "d_ffn": params["d_ffn"],
            "top_k": params["top_k"],
            "n_heads": params.get("n_heads", 8),
            "n_kernels": params.get("n_kernels", 6),
            "dropout": params["dropout"],
            "channel_mixing": params.get("channel_mixing", True),
            "channel_independence": params.get("channel_independence", False),
            "downsampling_layers": params.get("downsampling_layers", 3),
            "downsampling_window": params.get("downsampling_window", 2),
            "apply_nonstationary_norm": params.get("apply_nonstationary_norm", False),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define TimeMixer hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "d_model": trial.suggest_categorical("d_model", [64, 128]),
                "d_ffn": trial.suggest_categorical("d_ffn", [128, 256]),
                "top_k": trial.suggest_int("top_k", 3, 5),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8]),
                "n_kernels": trial.suggest_int("n_kernels", 4, 6),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "channel_mixing": trial.suggest_categorical("channel_mixing", [True, False]),
                "channel_independence": trial.suggest_categorical("channel_independence", [True, False]),
                "downsampling_layers": trial.suggest_int("downsampling_layers", 1, 3),
                "downsampling_window": trial.suggest_int("downsampling_window", 2, 3),
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 50, 150, step=50),
                "patience": trial.suggest_int("patience", 10, 20, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_layers": trial.suggest_int("n_layers", 2, 6),
                "d_model": trial.suggest_categorical("d_model", [64, 128, 256, 512]),
                "d_ffn": trial.suggest_categorical("d_ffn", [128, 256, 512, 1024]),
                "top_k": trial.suggest_int("top_k", 3, 7),
                "n_heads": trial.suggest_categorical("n_heads", [4, 8, 16]),
                "n_kernels": trial.suggest_int("n_kernels", 4, 8),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "channel_mixing": trial.suggest_categorical("channel_mixing", [True, False]),
                "channel_independence": trial.suggest_categorical("channel_independence", [True, False]),
                "downsampling_layers": trial.suggest_int("downsampling_layers", 1, 4),
                "downsampling_window": trial.suggest_int("downsampling_window", 2, 4),
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use TimeMixerImputer class instead"""
    imputer = TimeMixerImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use TimeMixerImputer class instead"""
    imputer = TimeMixerImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/timemixer.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run TimeMixer imputer
    imputer = TimeMixerImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()