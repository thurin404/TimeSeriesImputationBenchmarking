import os
import sys
import numpy as np
from typing import Dict, Any, Type
from pypots.imputation import BRITS
from pypots.nn.functional import calc_mae

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class BRITSImputer(BasePypotsImputer):
    """BRITS imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("brits")
    
    def get_model_class(self) -> Type:
        """Return the BRITS model class"""
        return BRITS
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return BRITS-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "rnn_hidden_size": params["rnn_hidden_size"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params.get("patience", 20),
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define BRITS hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            return {
                "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [64, 128, 256]),
                "batch_size": trial.suggest_categorical("batch_size", [16, 32]),
                "epochs": trial.suggest_int("epochs", 50, 200, step=50),
                "patience": trial.suggest_int("patience", 10, 20, step=5),
                "device": "cuda"  # Keep device fixed for consistency
            }
        else:
            # Standard parameters for normal memory environments
            return {
                "rnn_hidden_size": trial.suggest_categorical("rnn_hidden_size", [64, 128, 256, 384, 512]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 96, 128]),
                "epochs": trial.suggest_int("epochs", 50, 300, step=50),
                "patience": trial.suggest_int("patience", 10, 30, step=5),
                "device": "cuda"  # Keep device fixed for consistency
            }
    
    def prepare_data(self, df):
        """Override to add BRITS-specific logging"""
        print("Not enough complete samples to train BRITS model. Training on all available data with missing values.")
        return super().prepare_data(df)
    
    def evaluate_and_save_model(self, model, test_set, test_df_org, indicating_mask, model_dir):
        """Override to add BRITS-specific logging"""
        mae = calc_mae(model.impute(test_set), np.nan_to_num(test_df_org), indicating_mask)
        print(f"MAE on the validation set: {mae}")
        
        model_path = os.path.join(model_dir, f"{self.imputation_method.upper()}_{self.partition_by}.pypots")
        print(f"Saving the trained model to {model_path}")
        model.save(model_path, overwrite=True)
        self.model_trained = True
        
        return model


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use BRITSImputer class instead"""
    imputer = BRITSImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use BRITSImputer class instead"""
    imputer = BRITSImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/brits.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run BRITS imputer
    imputer = BRITSImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()
