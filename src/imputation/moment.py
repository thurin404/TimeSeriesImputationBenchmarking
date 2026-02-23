import sys
import os
from typing import Dict, Any, Type
try:
    from pypots.imputation import MOMENT
except ImportError:
    print("MOMENT not available in current PyPOTS version")
    MOMENT = None

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class MOMENTImputer(BasePypotsImputer):
    """MOMENT (A Family of Open Time-series Foundation Models) imputation implementation"""
    
    def __init__(self):
        super().__init__("moment")
        # MOMENT can use moderate memory
        self._env_overrides["low_memory_mode"] = False
    
    def get_model_class(self) -> Type:
        """Return the MOMENT model class"""
        if MOMENT is None:
            raise ImportError("MOMENT not available in current PyPOTS version")
        return MOMENT
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return MOMENT-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "patch_size": params["patch_size"],
            "patch_stride": params["patch_stride"],
            "transformer_backbone": params["transformer_backbone"],
            "transformer_type": params["transformer_type"],
            "n_layers": params["n_layers"],
            "d_ffn": params["d_ffn"],
            "d_model": params["d_model"],
            "dropout": params["dropout"],
            "head_dropout": params["head_dropout"],
            "finetuning_mode": params["finetuning_mode"],
            "revin_affine": params.get("revin_affine", True),
            "add_positional_embedding": params.get("add_positional_embedding", True),
            "value_embedding_bias": params.get("value_embedding_bias", True),
            "orth_gain": params.get("orth_gain", 1.41),
            "mask_ratio": params.get("mask_ratio", 0.3),
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define MOMENT hyperparameter search space for Optuna optimization
        
        Note: 
        - patch_size must divide evenly into n_steps
        - d_model must match the transformer backbone's hidden dimension:
          * flan-t5-small: d_model=512, d_ff=1024
          * flan-t5-base: d_model=768, d_ff=2048
        """
        if low_memory_mode:
            # Conservative parameters for low memory environments
            patch_size = trial.suggest_categorical("patch_size", [2, 4])
            patch_stride = patch_size  # Set equal to avoid warnings
            
            # Always use flan-t5-small for low memory
            # d_model and d_ffn must match T5 architecture
            return {
                "patch_size": patch_size,
                "patch_stride": patch_stride,
                "transformer_backbone": "google/flan-t5-small",
                "transformer_type": trial.suggest_categorical("transformer_type", ["encoder_only"]),
                "n_layers": trial.suggest_int("n_layers", 2, 4),
                "d_ffn": 1024,  # Fixed for flan-t5-small
                "d_model": 512,  # Fixed for flan-t5-small
                "dropout": trial.suggest_float("dropout", 0.0, 0.2, step=0.1),
                "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.2, step=0.1),
                "finetuning_mode": trial.suggest_categorical("finetuning_mode", ["end-to-end", "linear-probing"]),
                "revin_affine": True,
                "add_positional_embedding": True,
                "value_embedding_bias": True,
                "orth_gain": 1.41,
                "mask_ratio": trial.suggest_float("mask_ratio", 0.2, 0.4, step=0.1),
                "batch_size": trial.suggest_categorical("batch_size", [8, 16]),
                "epochs": trial.suggest_int("epochs", 25, 50, step=25),
                "patience": trial.suggest_int("patience", 5, 10, step=5)
            }
        else:
            # Standard parameters for normal memory environments
            patch_size = trial.suggest_categorical("patch_size", [2, 4])
            patch_stride = patch_size  # Set equal to avoid warnings
            
            # d_model and d_ffn must match the selected transformer backbone
            transformer_backbone = trial.suggest_categorical("transformer_backbone", 
                ["google/flan-t5-small", "t5-small", "t5-3b"])
            
            # Set d_model and d_ffn based on backbone
            if "small" in transformer_backbone:
                d_model, d_ffn = 512, 1024
                elif "t5-3b" in transformer_backbone:
                d_model, d_ffn = 1024, 512
            else:  # base
                d_model, d_ffn = 768, 2048
            
            return {
                "patch_size": patch_size,
                "patch_stride": patch_stride,
                "transformer_backbone": transformer_backbone,
                "transformer_type": trial.suggest_categorical("transformer_type", ["encoder_only", "encoder_decoder"]),
                "n_layers": trial.suggest_int("n_layers", 2, 8),
                "d_ffn": d_ffn,  # Fixed based on transformer backbone
                "d_model": d_model,  # Fixed based on transformer backbone
                "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.1),
                "head_dropout": trial.suggest_float("head_dropout", 0.0, 0.3, step=0.1),
                "finetuning_mode": trial.suggest_categorical("finetuning_mode", 
                    ["end-to-end", "linear-probing", "zero-shot"]),
                "revin_affine": trial.suggest_categorical("revin_affine", [True, False]),
                "add_positional_embedding": trial.suggest_categorical("add_positional_embedding", [True, False]),
                "value_embedding_bias": trial.suggest_categorical("value_embedding_bias", [True, False]),
                "orth_gain": trial.suggest_float("orth_gain", 1.0, 2.0, step=0.2),
                "mask_ratio": trial.suggest_float("mask_ratio", 0.2, 0.5, step=0.1),
                "batch_size": trial.suggest_categorical("batch_size", [2, 4, 6, 8]),
                "epochs": trial.suggest_int("epochs", 400, 500, step=100),
                "patience": trial.suggest_int("patience", 20, 30, step=10)
            }


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use MOMENTImputer class instead"""
    imputer = MOMENTImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use MOMENTImputer class instead"""
    imputer = MOMENTImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/moment.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run MOMENT imputer
    imputer = MOMENTImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()
