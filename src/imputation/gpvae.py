import sys
import os
from typing import Dict, Any, Type
from pypots.imputation import GPVAE

# Handle both direct execution and package imports
if __name__ == "__main__":
    # Add the src directory to the path for direct execution
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from imputation.base_pypots_imputer import BasePypotsImputer
else:
    # Package import
    from .base_pypots_imputer import BasePypotsImputer


class GPVAEImputer(BasePypotsImputer):
    """GPVAE imputation implementation using the base PyPOTS imputer"""
    
    def __init__(self):
        super().__init__("gpvae")
    
    def get_model_class(self) -> Type:
        """Return the GPVAE model class"""
        return GPVAE
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return GPVAE-specific parameters for initialization"""
        # Convert encoder_sizes to tuple (handles lists, tuples, and string representations)
        encoder_sizes = params["encoder_sizes"]
        if isinstance(encoder_sizes, str):
            # Convert string representation like "128_128" to tuple
            encoder_sizes = tuple(map(int, encoder_sizes.split("_")))
        elif isinstance(encoder_sizes, list):
            encoder_sizes = tuple(encoder_sizes)
            
        # Convert decoder_sizes to tuple (handles lists, tuples, and string representations)
        decoder_sizes = params["decoder_sizes"]
        if isinstance(decoder_sizes, str):
            # Convert string representation like "512_512" to tuple
            decoder_sizes = tuple(map(int, decoder_sizes.split("_")))
        elif isinstance(decoder_sizes, list):
            decoder_sizes = tuple(decoder_sizes)
        
        # Ensure numeric parameters are properly typed
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            "latent_size": int(params["latent_size"]),
            "encoder_sizes": encoder_sizes,
            "decoder_sizes": decoder_sizes,
            "kernel": params["kernel"],
            "beta": float(params["beta"]),
            "M": int(params["M"]),
            "K": int(params["K"]),
            "sigma": float(params["sigma"]),
            "length_scale": float(params["length_scale"]) if not isinstance(params["length_scale"], (list, tuple)) else params["length_scale"],
            "kernel_scales": int(params.get("kernel_scales", 1)),
            "window_size": int(params.get("window_size", 3)),
            "batch_size": int(params["batch_size"]),
            "epochs": int(params["epochs"]),
            "patience": int(params["patience"]),
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define GPVAE hyperparameter search space for Optuna optimization"""
        if low_memory_mode:
            # Conservative parameters for low memory environments
            encoder_sizes_str = trial.suggest_categorical("encoder_sizes", ["64_64", "128_128"])
            decoder_sizes_str = trial.suggest_categorical("decoder_sizes", ["64_64", "128_128"])
        else:
            # Standard parameters for normal memory environments
            encoder_sizes_str = trial.suggest_categorical("encoder_sizes", ["64_64", "128_128", "256_256", "512_512"])
            decoder_sizes_str = trial.suggest_categorical("decoder_sizes", ["64_64", "128_128", "256_256", "512_512"])

        # Convert string representation back to tuples
        encoder_sizes = tuple(map(int, encoder_sizes_str.split("_")))
        decoder_sizes = tuple(map(int, decoder_sizes_str.split("_")))
        
        base_params = {
            "latent_size": trial.suggest_int("latent_size", 16, 32 if low_memory_mode else 64, step=1),
            "encoder_sizes": encoder_sizes,
            "decoder_sizes": decoder_sizes,
            "kernel": trial.suggest_categorical("kernel", ["cauchy", "rbf", "matern"]),
            "beta": trial.suggest_float("beta", 0.1, 1.0),
            "M": trial.suggest_int("M", 1, 3 if low_memory_mode else 5),
            "K": trial.suggest_int("K", 1, 3 if low_memory_mode else 5),
            "sigma": trial.suggest_float("sigma", 0.9, 1.5),
            "length_scale": trial.suggest_categorical("length_scale", [4, 8, 12, 16]),
            "kernel_scales": trial.suggest_int("kernel_scales", 1, 2 if low_memory_mode else 3),
            "window_size": trial.suggest_int("window_size", 3, 5 if low_memory_mode else 50),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32] if low_memory_mode else [16, 32, 64])
        }
        
        return base_params


# Legacy functions for backward compatibility (deprecated)
def imputation(df, params, model_dir):
    """Deprecated: Use GPVAEImputer class instead"""
    imputer = GPVAEImputer()
    return imputer.imputation(df, params, model_dir)


def training(df, params, model_dir):
    """Deprecated: Use GPVAEImputer class instead"""
    imputer = GPVAEImputer()
    return imputer.training(df, params, model_dir)


def main():
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/gpvae.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run GPVAE imputer
    imputer = GPVAEImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()