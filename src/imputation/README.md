# PyPOTS Imputation Architecture

This directory contains a refactored architecture for PyPOTS-based time series imputation models. The architecture uses object-oriented design to make it easy to add new models and apply structural changes across all implementations.

## Architecture Overview

### Base Class: `BasePypotsImputer`

The `BasePypotsImputer` class provides common functionality for all PyPOTS imputation models:

- **Data preparation**: Handles reshaping, train/val/test splits, and missing value masking
- **Model lifecycle**: Training, loading, saving, and evaluation
- **Pipeline orchestration**: Complete workflow from data loading to result saving
- **Configuration management**: YAML parameter loading and model state tracking

### Implemented Models

#### 1. SAITS (Self-Attention-based Imputation for Time Series)
- **File**: `saits.py`
- **Class**: `SAITSImputer`
- **Parameters**: Transformer-based architecture with attention mechanisms

#### 2. BRITS (Bidirectional Recurrent Imputation for Time Series)
- **File**: `brits.py`
- **Class**: `BRITSImputer`
- **Parameters**: RNN-based architecture with bidirectional processing
- **Special features**: Custom logging for training on incomplete data

## Adding New Models

To add a new PyPOTS model, follow these steps:

### 1. Create a New Imputer Class

```python
from pypots.imputation import YourNewModel
from base_pypots_imputer import BasePypotsImputer

class YourNewModelImputer(BasePypotsImputer):
    def __init__(self):
        super().__init__("your_model_name")
    
    def get_model_class(self):
        return YourNewModel
    
    def get_model_params(self, params, n_steps, n_features, model_dir):
        return {
            "n_steps": n_steps,
            "n_features": n_features,
            # Add model-specific parameters here
            "batch_size": params["batch_size"],
            "epochs": params["epochs"],
            "saving_path": model_dir,
            "model_saving_strategy": "best"
        }
```

### 2. Required Methods

Every new imputer must implement:

- `get_model_class()`: Returns the PyPOTS model class
- `get_model_params()`: Returns model initialization parameters

### 3. Optional Method Overrides

You can override these methods for model-specific behavior:

- `prepare_data()`: Custom data preparation
- `evaluate_and_save_model()`: Custom evaluation or logging
- `load_or_train_model()`: Custom training logic

### 4. Create the Main Script

```python
def main():
    if len(sys.argv) != 4:
        print("Usage: python your_model.py <input_dir> <output_dir> <model_dir>")
        sys.exit(1)

    input_dir, output_dir, model_dir = sys.argv[1:4]
    
    imputer = YourNewModelImputer()
    imputer.run_pipeline(input_dir, output_dir, model_dir)

if __name__ == "__main__":
    main()
```

### 5. Configuration

Add your model configuration to the appropriate YAML files:
- Update `params.yaml` with model references
- Create model-specific parameter files in `params/`

## Benefits of This Architecture

1. **Code Reuse**: Common functionality is centralized in the base class
2. **Consistency**: All models follow the same patterns and interfaces
3. **Maintainability**: Changes to common logic apply to all models
4. **Extensibility**: New models require minimal boilerplate code
5. **Backward Compatibility**: Legacy function interfaces are preserved

## Usage Examples

### Using the Class-Based Interface

```python
from saits import SAITSImputer

imputer = SAITSImputer()
imputed_df = imputer.imputation(df, params, model_dir)
```

### Using the Legacy Interface (Deprecated)

```python
from saits import imputation

imputed_df = imputation(df, params, model_dir)
```

### Running from Command Line

```bash
python src/imputation/saits.py data/04_working_files data/05_imputed models/saits
python src/imputation/brits.py data/04_working_files data/05_imputed models/brits
```

## File Structure

```
src/imputation/
├── base_pypots_imputer.py      # Base class with common functionality
├── saits.py                    # SAITS implementation
├── brits.py                    # BRITS implementation
├── new_model_template.py       # Template for new models
└── README.md                   # This documentation
```

## Migration Notes

The refactored scripts maintain backward compatibility with existing code:
- Legacy functions (`imputation()`, `training()`) are preserved but deprecated
- Command-line interfaces remain unchanged
- Configuration file formats are unchanged

This ensures existing workflows continue to work while new development can leverage the improved architecture.