import os
import shutil
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, Optional
from .tools import utilities
import pandas as pd
import numpy as np
from pygrinder import mcar
from pypots.nn.functional import calc_mae


try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import HyperbandPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not available. Hyperparameter optimization will be disabled.")


class BasePypotsImputer(ABC):
    """
    Base class for PyPOTS imputation models.
    Provides common functionality for training, loading, and using PyPOTS models.
    """
    
    def __init__(self, imputation_method: str):
        self.imputation_method = imputation_method
        self.model_trained = False
        self.partition_by = None
        
        # Environment variable overrides for key parameters
        self._env_overrides = self._load_env_overrides()
        
    def _load_env_overrides(self) -> Dict[str, Any]:
        """Load environment variable overrides for optimization and training parameters"""
        overrides = {}
        
        # Optimization parameters
        if os.getenv("OPTUNA_N_TRIALS"):
            try:
                overrides["n_trials"] = int(os.getenv("OPTUNA_N_TRIALS", "0"))
            except ValueError:
                print(f"Warning: Invalid OPTUNA_N_TRIALS value: {os.getenv('OPTUNA_N_TRIALS')}")
        
        if os.getenv("OPTUNA_TIMEOUT"):
            try:
                overrides["timeout"] = int(os.getenv("OPTUNA_TIMEOUT", "0"))
            except ValueError:
                print(f"Warning: Invalid OPTUNA_TIMEOUT value: {os.getenv('OPTUNA_TIMEOUT')}")
        
        if os.getenv("OPT_EPOCH_FRACTION"):
            try:
                overrides["opt_epoch_fraction"] = float(os.getenv("OPT_EPOCH_FRACTION", "0.0"))
            except ValueError:
                print(f"Warning: Invalid OPT_EPOCH_FRACTION value: {os.getenv('OPT_EPOCH_FRACTION')}")
        
        if os.getenv("TEST_RUN_EPOCH_FRACTION"):
            try:
                overrides["test_run_epoch_fraction"] = float(os.getenv("TEST_RUN_EPOCH_FRACTION", "0.0"))
            except ValueError:
                print(f"Warning: Invalid TEST_RUN_EPOCH_FRACTION value: {os.getenv('TEST_RUN_EPOCH_FRACTION')}")
        
        # Model training parameters
        if os.getenv("MAX_EPOCHS"):
            try:
                overrides["max_epochs"] = int(os.getenv("MAX_EPOCHS", "0"))
            except ValueError:
                print(f"Warning: Invalid MAX_EPOCHS value: {os.getenv('MAX_EPOCHS')}")
        
        if os.getenv("BATCH_SIZE"):
            try:
                overrides["batch_size"] = int(os.getenv("BATCH_SIZE", "0"))
            except ValueError:
                print(f"Warning: Invalid BATCH_SIZE value: {os.getenv('BATCH_SIZE')}")
        
        # Memory optimization
        low_mem_env = os.getenv("LOW_MEMORY_MODE")
        if low_mem_env:
            overrides["low_memory_mode"] = low_mem_env.lower() in ("1", "true", "yes")
        
        return overrides
        
    @abstractmethod
    def get_model_class(self) -> Type:
        """Return the PyPOTS model class (e.g., SAITS, BRITS)"""
        pass
    
    @abstractmethod
    def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str, is_optimizing: bool = False) -> Dict[str, Any]:
        """Return model-specific parameters for initialization
        
        Args:
            params: Model parameters
            n_steps: Number of time steps
            n_features: Number of features
            model_dir: Directory for saving models
            is_optimizing: Whether we're in hyperparameter optimization mode (don't save models)
        """
        pass
    
    @abstractmethod
    def get_optimization_params(self, trial, low_memory_mode: bool = False) -> Dict[str, Any]:
        """Define hyperparameter search space for optimization"""
        pass
    
    def optimize_hyperparameters(self, df: pd.DataFrame, model_dir: str, 
                                n_trials: int = 50, timeout: int = 3600,
                                opt_epoch_fraction: float = 0.25,
                                base_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            df: Input dataframe for optimization
            model_dir: Model directory path
            n_trials: Number of optimization trials
            timeout: Timeout in seconds for optimization
            opt_epoch_fraction: Fraction to scale epochs during optimization
            base_params: Base parameters (fixed params like epochs, patience) to merge with trial params
            
        Returns:
            Best parameters found
        """
        if not OPTUNA_AVAILABLE:
            print("Optuna not available. Skipping hyperparameter optimization.")
            return {}
        
        if base_params is None:
            base_params = {}
        
        # Prepare data for optimization
        train_set, val_set, test_set, test_df_org, indicating_mask = self.prepare_data(df)
        
        def objective(trial):
            try:
                # Get low memory mode setting
                low_memory_mode = self._env_overrides.get("low_memory_mode", False)
                
                # Get trial parameters
                params = self.get_optimization_params(trial, low_memory_mode=low_memory_mode)
                
                # Merge with base parameters (fixed params like epochs, patience)
                params = {**base_params, **params}

                # Scale epochs and patience during optimization to save time
                if "epochs" in params:
                    original_epochs = params["epochs"]
                    scaled_epochs = max(1, int(original_epochs * opt_epoch_fraction))
                    params["epochs"] = scaled_epochs
                if "patience" in params:
                    original_patience = params["patience"]
                    scaled_patience = max(1, int(original_patience * opt_epoch_fraction))
                    params["patience"] = scaled_patience

                # Inform about scaled settings
                print(f"Optimization trial using epochs={params.get('epochs')} (scaled from {original_epochs if 'original_epochs' in locals() else 'N/A'}) and patience={params.get('patience')} ")

                # Create model with trial parameters (don't save during optimization)
                model = self.create_model(df, params, model_dir, is_optimizing=True)

                # Train model (without loading existing model)
                model.fit(train_set, val_set)

                # Evaluate model
                predictions = model.impute(test_set)
                
                # Handle extra dimensions that some models (like GPVAE) might return
                # Expected shape: (n_samples, n_steps, n_features)
                # Some models return: (n_samples, 1, n_steps, n_features)
                if predictions.ndim == 4 and predictions.shape[1] == 1:
                    predictions = predictions.squeeze(1)
                
                mae = calc_mae(predictions, np.nan_to_num(test_df_org), indicating_mask)

                # Clean up model to save memory
                del model

                return mae

            except Exception as e:
                print(f"Trial failed with error: {str(e)}")
                return float('inf')  # Return worst possible score for failed trials
        
        # Create study with sophisticated sampler and pruner
        sampler = TPESampler(
            n_startup_trials=10,
            n_ei_candidates=24,
            multivariate=True,
            seed=42
        )
        
        pruner = HyperbandPruner(
            min_resource=5,
            max_resource=50,
            reduction_factor=3
        )
        
        study = optuna.create_study(
            direction='minimize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"{self.imputation_method}_optimization"
        )
        
        # Optimize
        print(f"Starting hyperparameter optimization for {self.imputation_method}...")
        print(f"Running {n_trials} trials with {timeout} seconds timeout")
        
        study.optimize(
            objective, # type: ignore
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        # Get best parameters
        best_params = study.best_params
        best_value = study.best_value
        
        print("Optimization completed!")
        print(f"Best MAE: {best_value:.6f}")
        print(f"Best parameters: {best_params}")
        
        # Generate optimization report
        self._save_optimization_report(study, model_dir)
        
        return best_params
    
    def _save_optimization_report(self, study, model_dir: str):
        """Save optimization report and visualizations"""
        if not OPTUNA_AVAILABLE:
            return
            
        try:
            # Create optimization report directory
            report_dir = os.path.join(model_dir, f"{self.imputation_method}_optimization")
            os.makedirs(report_dir, exist_ok=True)
            
            # Save study statistics
            stats = {
                'best_value': study.best_value,
                'best_params': study.best_params,
                'n_trials': len(study.trials),
                'datetime_start': getattr(study, 'datetime_start', None),
                'datetime_complete': getattr(study, 'datetime_complete', None),
                'study_name': getattr(study, 'study_name', f"{self.imputation_method}_optimization")
            }
            
            # Convert datetime objects to ISO format if they exist
            if stats['datetime_start']:
                stats['datetime_start'] = stats['datetime_start'].isoformat()
            if stats['datetime_complete']:
                stats['datetime_complete'] = stats['datetime_complete'].isoformat()
            
            with open(os.path.join(report_dir, 'optimization_stats.yaml'), 'w') as f:
                yaml.dump(stats, f, default_flow_style=False)
            
            # Save trials dataframe
            trials_df = study.trials_dataframe()
            trials_df.to_csv(os.path.join(report_dir, 'trials.csv'), index=False)
            
            print(f"Optimization report saved to: {report_dir}")
            
        except Exception as e:
            print(f"Failed to save optimization report: {str(e)}")
    
    def update_params_with_optimization(self, model_config_path: str, best_params: Dict[str, Any]):
        """Update the model configuration file with optimized parameters"""
        if not best_params:
            return
            
        try:
            from config_manager import ModelConfigManager
            config_manager = ModelConfigManager(model_config_path)
            
            # Prepare optimization info
            optimization_info = {
                "method": "optuna_tpe_hyperband",
                "model_type": self.imputation_method
            }
            
            # Update configuration
            config_manager.update_with_optimization(
                best_params, 
                self.partition_by,
                optimization_info
            )
            
        except Exception as e:
            print(f"Failed to update model configuration: {str(e)}")
            # Fallback to original method
            try:
                with open(model_config_path, "r") as f:
                    model_yaml = yaml.safe_load(f)
                
                # Update parameters with optimized values
                for param, value in best_params.items():
                    if param in model_yaml["model"]:
                        # Convert tuples to lists for YAML safety
                        if isinstance(value, tuple):
                            value = list(value)
                        model_yaml["model"][param] = value
                        print(f"Updated {param}: {model_yaml['model'][param]} -> {value}")
                
                # Mark as optimized
                model_yaml["optimized"] = True
                
                # Save updated configuration
                with open(model_config_path, "w") as f:
                    yaml.dump(model_yaml, f, default_flow_style=False)
                
                print(f"Updated model configuration saved to: {model_config_path}")
                
            except Exception as e2:
                print(f"Fallback update also failed: {str(e2)}")
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """Prepare data for training and validation"""
    
        if len(df) % 96 != 0:
            n = len(df) - (len(df) % 96)
            print(f"Training model on {n} samples.")
            df = df.iloc[:n]
            df_missing_val = df.values.reshape(-1,96,df.shape[1])
        else:
            df_missing_val = df.values.reshape(-1,96,df.shape[1])

        #df_missing_val = df.values.reshape(-1, 96, df.shape[1])
        train_df = df_missing_val[:int(len(df_missing_val) * 0.7)]
        val_df_org = df_missing_val[int(len(df_missing_val) * 0.7):int(len(df_missing_val) * 0.85)]
        test_df_org = df_missing_val[int(len(df_missing_val) * 0.85):]

        val_df = val_df_org.copy()
        test_df = test_df_org.copy()
        val_df = mcar(val_df, p=0.2)
        test_df = mcar(test_df, p=0.2)

        train_set = {"X": train_df}
        val_set = {
            "X": val_df,
            "X_ori": val_df_org,
        }
        test_set = {"X": test_df}
        
        indicating_mask = np.isnan(test_df) ^ np.isnan(test_df_org)
        
        return train_set, val_set, test_set, test_df_org, indicating_mask
    
    def create_model(self, df: pd.DataFrame, params: Dict[str, Any], model_dir: str, is_optimizing: bool = False):
        """Create and initialize the model
        
        Args:
            df: Input dataframe
            params: Model parameters
            model_dir: Directory for saving models
            is_optimizing: Whether we're in hyperparameter optimization mode (don't save models)
        """

        if len(df) % 96 != 0:
            missing_n: int = 96 - len(df) % 96
            filler = np.zeros((missing_n, df.shape[1])) * np.nan
            df = pd.concat([df, pd.DataFrame(filler, columns=df.columns)], ignore_index=True)
            print(f"Imputing on {len(df)} samples (added {missing_n} filler samples).")
        df_missing_val = df.values.reshape(-1, 96, df.shape[1])
        n_steps = df_missing_val.shape[1]
        n_features = df_missing_val.shape[2]
        
        model_class = self.get_model_class()
        model_params = self.get_model_params(params, n_steps, n_features, model_dir, is_optimizing=is_optimizing)
        
        return model_class(**model_params)
    
    def load_or_train_model(self, model, train_set: Dict, val_set: Dict, model_dir: str):
        """Load existing model or train new one"""
        model_path = os.path.join(model_dir, f"{self.imputation_method.upper()}_{self.partition_by}.pypots")
        if model is not None:
            original_epochs = model.epochs
            model.epochs = int(original_epochs / 4)
            
        if self.model_trained and os.path.exists(model_path):
            try:
                model.load(path=model_path)
            except Exception as e:
                # Loading failed (likely architecture mismatch). Train from scratch instead.
                print(f"Warning: Failed to load existing model (will train from scratch): {e}")
                model.fit(train_set, val_set)
        else:
            model.fit(train_set, val_set)
        
        # Continue training with reduced epochs
        if model is not None:
            if original_epochs:
                model.epochs = original_epochs
                model.fit(train_set, val_set)
            
        return model
    
    def evaluate_and_save_model(self, model, test_set: Dict, test_df_org: np.ndarray, 
                               indicating_mask: np.ndarray, model_dir: str):
        """Evaluate model performance and save it"""
        predictions = model.impute(test_set)
        
        # Handle extra dimensions that some models (like GPVAE) might return
        # Expected shape: (n_samples, n_steps, n_features)
        # Some models return: (n_samples, 1, n_steps, n_features)
        if predictions.ndim == 4 and predictions.shape[1] == 1:
            predictions = predictions.squeeze(1)
        
        mae = calc_mae(predictions, np.nan_to_num(test_df_org), indicating_mask)
        print(f"MAE on the validation set: {mae}")
        
        model_path = os.path.join(model_dir, f"{self.imputation_method.upper()}_{self.partition_by}.pypots")
        model.save(model_path, overwrite=True)
        self.model_trained = True
        
        return model
    
    def training(self, df: pd.DataFrame, params: Dict[str, Any], model_dir: str):
        """Complete training pipeline"""
        train_set, val_set, test_set, test_df_org, indicating_mask = self.prepare_data(df)
        model = self.create_model(df, params, model_dir)
        model = self.load_or_train_model(model, train_set, val_set, model_dir)
        model = self.evaluate_and_save_model(model, test_set, test_df_org, indicating_mask, model_dir)
        
        return model
    
    def imputation(self, df: pd.DataFrame, params: Dict[str, Any], model_dir: str) -> pd.DataFrame:
        """Perform imputation on the dataframe"""
        # Store original index before any modifications
        original_index = df.index.copy()
        
        model = self.training(df, params, model_dir)
        if len(df) % 96 != 0:
            missing_n: int = 96 - len(df) % 96
            filler = np.zeros((missing_n, df.shape[1])) * np.nan
            # Create filler with temporary indices (will be removed later)
            filler_df = pd.DataFrame(filler, columns=df.columns)
            df = pd.concat([df.reset_index(drop=True), filler_df], ignore_index=True)
            print(f"Imputing on {len(df)} samples (added {missing_n} filler samples).")
        else:
            # Even if no padding needed, reset index for consistent reshaping
            df = df.reset_index(drop=True)
            missing_n = 0
            
        df_missing_val = df.values.reshape(-1, 96, df.shape[1])
        test_set = {"X": df_missing_val}
        imputed_values = model.impute(test_set)
        
        # Handle extra dimensions that some models (like GPVAE) might return
        # Expected shape: (n_samples, n_steps, n_features)
        # Some models return: (n_samples, 1, n_steps, n_features)
        if imputed_values.ndim == 4 and imputed_values.shape[1] == 1:
            imputed_values = imputed_values.squeeze(1)
        
        df_imputed = pd.DataFrame(
            imputed_values.reshape(-1, df.shape[1]), 
            columns=df.columns
        )
        
        # Remove filler rows if they were added
        if missing_n > 0:
            df_imputed = df_imputed.iloc[:-missing_n]
        
        # Restore original index
        df_imputed.index = original_index
        
        return df_imputed
    
    def run_pipeline(self, input_dir: str, output_dir: str, model_dir: str):
        """Run the complete imputation pipeline"""
        # Load YAML configurations
        params_all = yaml.safe_load(open("params.yaml"))
        model_config_path = params_all["imputation"][self.imputation_method]["params_file"]
        self.partition_by = params_all["prepare_working_files"]["partition_by"]

        with open(model_config_path, "r") as f:
            model_yaml = yaml.safe_load(f)

        # Support both old and new parameter formats
        if "model" in model_yaml:
            model_params = model_yaml["model"]
            self.model_trained = model_yaml.get("model-trained", {}).get(self.partition_by.lower(), False)
            # New format: optimization_enabled = true means "should run optimization"
            # Convert to old logic where optimized = true means "already optimized"
            optimization_enabled = model_yaml.get("optimization", {}).get("optimization_enabled", False)
            optimized = not optimization_enabled  # Invert the logic
        else:
            # Legacy format support
            model_params = model_yaml["params"]
            self.model_trained = model_yaml["model-trained"][self.partition_by.lower()]
            optimized = model_yaml.get("optimized", False)

        if "optimized" in model_yaml:
            optimized = model_yaml.get("optimized", False)

        data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
        _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)

        imputed_data = {}

        # Handle optimization
        if not optimized:
            print(f"Parameters not optimized for {self.imputation_method}. Starting hyperparameter optimization...")

            if OPTUNA_AVAILABLE:
                # Use first dataset for optimization
                sample_key = next(iter(missing_data.keys()))
                sample_df = missing_data[sample_key]

                # Read optimization config (allow overriding n_trials/timeout/epoch fraction)
                opt_cfg = model_yaml.get("optimization", {})
                n_trials_cfg = self._env_overrides.get("n_trials", opt_cfg.get("n_trials", 50))
                timeout_cfg = self._env_overrides.get("timeout", opt_cfg.get("timeout", 3600))
                opt_epoch_fraction = self._env_overrides.get("opt_epoch_fraction", opt_cfg.get("opt_epoch_fraction", 0.25))

                # Run optimization (epochs used inside trials will be scaled by opt_epoch_fraction)
                # Pass model_params as base_params so fixed parameters (like epochs, patience) are available
                best_params = self.optimize_hyperparameters(
                    sample_df,
                    model_dir,
                    n_trials=n_trials_cfg,
                    timeout=timeout_cfg,
                    opt_epoch_fraction=opt_epoch_fraction,
                    base_params=model_params
                )

                # Update model configuration with optimized parameters
                if best_params:
                    self.update_params_with_optimization(model_config_path, best_params)
                    # Reload updated parameters
                    with open(model_config_path, "r") as f:
                        model_yaml = yaml.safe_load(f)
                    model_params = model_yaml["model"]
                    print("Using optimized parameters for imputation.")
                else:
                    print("Optimization failed. Using default parameters.")
            else:
                print("Optuna not available. Using default parameters.")
        else:
            print(f"Using pre-optimized parameters for {self.imputation_method} imputation.")

        # Check for a quick TEST_RUN (use env var TEST_RUN=1 or true)
        test_run_env = os.getenv("TEST_RUN", "0").lower()
        TEST_RUN = test_run_env in ("1", "true", "yes")
        test_epoch_fraction = self._env_overrides.get("test_run_epoch_fraction", 
                                                     model_yaml.get("optimization", {}).get("test_run_epoch_fraction", 0.1))

        # If TEST_RUN is active, automatically enable low memory mode to avoid OOM
        if TEST_RUN:
            self._env_overrides["low_memory_mode"] = True
            print("TEST_RUN active: automatically enabling low memory mode")

        # If a test run was requested, scale down epochs in model_params
        if TEST_RUN:
            if "epochs" in model_params:
                orig = model_params["epochs"]
                model_params["epochs"] = max(3, int(orig * test_epoch_fraction))  # Minimum 3 epochs for stability
                print(f"TEST_RUN active: scaling final epochs {orig} -> {model_params['epochs']}")
            if "patience" in model_params:
                origp = model_params["patience"]
                model_params["patience"] = max(2, int(origp * test_epoch_fraction))  # Minimum 2 for patience
                print(f"TEST_RUN active: scaling patience {origp} -> {model_params['patience']}")
            
            # Scale down memory-intensive parameters for ImputeFormer
            if self.imputation_method == "imputeformer":
                if "batch_size" in model_params and model_params["batch_size"] > 16:
                    model_params["batch_size"] = 16
                    print(f"TEST_RUN active: reducing batch_size to {model_params['batch_size']}")
                if "d_learnable_embed" in model_params and model_params["d_learnable_embed"] > 64:
                    model_params["d_learnable_embed"] = 64
                    print(f"TEST_RUN active: reducing d_learnable_embed to {model_params['d_learnable_embed']}")
                if "d_ffn" in model_params and model_params["d_ffn"] > 64:
                    model_params["d_ffn"] = 64
                    print(f"TEST_RUN active: reducing d_ffn to {model_params['d_ffn']}")
                if "n_layers" in model_params and model_params["n_layers"] > 2:
                    model_params["n_layers"] = 2
                    print(f"TEST_RUN active: reducing n_layers to {model_params['n_layers']}")
            
            # Scale down memory-intensive parameters for other models
            if "batch_size" in model_params and model_params["batch_size"] > 32:
                orig_batch = model_params["batch_size"]
                model_params["batch_size"] = 32
                print(f"TEST_RUN active: reducing batch_size {orig_batch} -> {model_params['batch_size']}")
        
        # Apply environment variable overrides to model params
        if "max_epochs" in self._env_overrides:
            model_params["epochs"] = self._env_overrides["max_epochs"]
            print(f"Environment override: epochs set to {model_params['epochs']}")
        
        if "batch_size" in self._env_overrides:
            model_params["batch_size"] = self._env_overrides["batch_size"]
            print(f"Environment override: batch_size set to {model_params['batch_size']}")

        # Perform imputation
        for key, df in missing_data.items():
            print(f"Imputing data for: {key} using {self.imputation_method} method")
            imputed_data[key] = self.imputation(df, model_params, model_dir)
            print(f"Imputation completed for: {key}")

        utilities.save_imputed_files(imputed_data, output_dir=output_dir,
                                   imputation_method=self.imputation_method, prep_method=prep_method)

        # Update model training status
        model_yaml["model-trained"][self.partition_by.lower()] = self.model_trained
        with open(model_config_path, "w") as f:
            yaml.dump(model_yaml, f)

        # Copy original files
        self._copy_original_files(data_files, input_dir, output_dir)
    
    def _copy_original_files(self, data_files: list, input_dir: str, output_dir: str):
        """Copy original files to output directory"""
        for filepath in data_files:
            filename = os.path.basename(filepath)
            src_path = os.path.join(input_dir, filename)
            dest_path = os.path.join(output_dir, filename)

            if not os.path.exists(dest_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied: {filename}")
            else:
                print(f"Already exists: {filename}")