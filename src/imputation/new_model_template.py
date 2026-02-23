"""
Example template for creating a new PyPOTS imputation model.
This template shows how to easily add new models using the BasePypotsImputer class.
"""

import sys
from typing import Dict, Any, Type
# from pypots.imputation import YourNewModel  # Import your new model here
from base_pypots_imputer import BasePypotsImputer


class NewModelImputer(BasePypotsImputer):
    """
    Template for implementing a new PyPOTS model.
    Replace 'NewModel' with your actual model name (e.g., GPVAEImputer, TransformerImputer, etc.)
    """
    
    def __init__(self):
        super().__init__("new_model")  # Replace with your model name
    
    def get_model_class(self) -> Type:
        """Return the PyPOTS model class"""
        # return YourNewModel  # Replace with your actual model class
        raise NotImplementedError("Replace with your actual model class")
    
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return model-specific parameters for initialization"""
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            # Add your model-specific parameters here
            # Example:
            # "hidden_size": params["hidden_size"],
            # "num_layers": params["num_layers"],
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "patience": params["patience"],
            "saving_path": model_dir,
            "model_saving_strategy": None if is_optimizing else "best"
        }
    
    # Optional: Override methods for model-specific behavior
    # def prepare_data(self, df):
    #     """Override to add model-specific data preparation"""
    #     # Add custom logging or data preparation here
    #     return super().prepare_data(df)
    #
    # def evaluate_and_save_model(self, model, test_set, test_df_org, indicating_mask, model_dir):
    #     """Override to add model-specific evaluation or saving logic"""
    #     # Add custom evaluation or logging here
    #     return super().evaluate_and_save_model(model, test_set, test_df_org, indicating_mask, model_dir)


def main():
    """Main function to run the imputation pipeline"""
    if len(sys.argv) != 4:
        print("Usage: python src/imputation/new_model.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    model_dir = sys.argv[3]

    # Create and run the imputer
    imputer = NewModelImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)


if __name__ == "__main__":
    main()