import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import MissNet
except ImportError:
    print("MissNet not available in current PyPOTS version")
    MissNet = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class MissNetImputer(BasePypotsImputer):
    """MissNet imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("missnet")
    
    def get_model_class(self) -> Type:
        """Return the MissNet model class"""
        if MissNet is None:
            raise ImportError("MissNet not available in current PyPOTS version")
        return MissNet
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return MissNet-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "rnn_hidden_size": params["rnn_hidden_size"],
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define MissNet hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [64, 128, 256]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 50, 150, step=50),
                "patience": trial.suggest_int("patience", 10, 20, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [64, 128, 256, 384, 512]),
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use MissNetImputer class instead"""
    imputer = MissNetImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use MissNetImputer class instead"""
    imputer = MissNetImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/missnet.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run MissNet imputer
    imputer = MissNetImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()