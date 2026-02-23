import sys
import os
from typing import Dict, Any, Type
from pypots.imputation import TEFN

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class TEFNImputer(BasePypotsImputer):
    """TEFN imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("tefn")
    
    def get_model_class(self) -> Type:
        """Return the TEFN model class"""
        return TEFN
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return TEFN-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "n_fod": params.get("n_fod", 2),
            "apply_nonstationary_norm": params.get("apply_nonstationary_norm", True),
            "ORT_weight": params.get("ORT_weight", 1.0),
            "MIT_weight": params.get("MIT_weight", 1.0),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define TEFN hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "n_fod": trial.suggest_int("n_fod", 1, 3),
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 2.0),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 2.0),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 50, 150, step=50),
                "patience": trial.suggest_int("patience", 10, 20, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "n_fod": trial.suggest_int("n_fod", 4, 10),
                "apply_nonstationary_norm": trial.suggest_categorical("apply_nonstationary_norm", [True, False]),
                "ORT_weight": trial.suggest_float("ORT_weight", 0.5, 1.5),
                "MIT_weight": trial.suggest_float("MIT_weight", 0.5, 1.5),
                "batch_size": trial.suggest_categorical("batch_size", [64, 96]),
                "epochs": trial.suggest_int("epochs", 400, 500, step=100),
                "patience": trial.suggest_int("patience", 10, 20, step=10)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use TEFNImputer class instead"""
    imputer = TEFNImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use TEFNImputer class instead"""
    imputer = TEFNImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/tefn.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run TEFN imputer
    imputer = TEFNImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()