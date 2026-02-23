#!/usr/bin/env python3
"""
Example script demonstrating the use of the PyPOTS imputation framework
with sophisticated hyperparameter optimization.

This script shows how to:
1. Create imputer instances
2. Run hyperparameter optimization
3. Use optimized parameters for imputation
4. Manage configuration files

Usage:
    python example_optimization.py
"""

import os
import sys
from pathlib import Path

# Add the imputation directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from saits import SAITSImputer
from brits import BRITSImputer
from config_manager import load_model_config
from tools import utilities


def demonstrate_optimization():
    """Demonstrate the optimization workflow"""
    
    print("=" * 60)
    print("PyPOTS Imputation with Hyperparameter Optimization Demo")
    print("=" * 60)
    
    # Configuration
    input_dir = "../../data/04_working_files"
    model_dir = "../../models"
    
    # Check if data directory exists
    if not os.path.exists(input_dir):
        print(f"Data directory not found: {input_dir}")
        print("Please run the data preparation pipeline first.")
        return
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Load data
    print("\n1. Loading data...")
    try:
        data_files, missing_data_files = utilities.get_working_files(dir=input_dir)
        _, missing_data, prep_method = utilities.load_working_files(data_files, missing_data_files)
        
        if not missing_data:
            print("No data found!")
            return
        
        # Use first dataset for demonstration
        sample_key = next(iter(missing_data.keys()))
        sample_df = missing_data[sample_key]
        
        print(f"   ✓ Loaded dataset: {sample_key}")
        print(f"   ✓ Dataset shape: {sample_df.shape}")
        print(f"   ✓ Preparation method: {prep_method}")
        
    except Exception as e:
        print(f"   ✗ Error loading data: {e}")
        return
    
    # Demonstrate SAITS optimization
    print("\n2. Demonstrating SAITS optimization...")
    try:
        saits_imputer = SAITSImputer()
        print("   ✓ Created SAITS imputer")
        
        # Run optimization (small number of trials for demo)
        saits_model_dir = os.path.join(model_dir, "saits_demo")
        os.makedirs(saits_model_dir, exist_ok=True)
        
        print("   → Running hyperparameter optimization (this may take a while)...")
        best_params = saits_imputer.optimize_hyperparameters(
            sample_df,
            saits_model_dir,
            n_trials=10,  # Small number for demo
            timeout=300   # 5 minutes timeout
        )
        
        if best_params:
            print(f"   ✓ Optimization completed!")
            print(f"   ✓ Best parameters: {best_params}")
        else:
            print("   ✗ Optimization failed")
            
    except Exception as e:
        print(f"   ✗ Error during SAITS optimization: {e}")
    
    # Demonstrate BRITS optimization
    print("\n3. Demonstrating BRITS optimization...")
    try:
        brits_imputer = BRITSImputer()
        print(f"   ✓ Created BRITS imputer")
        
        # Run optimization (small number of trials for demo)
        brits_model_dir = os.path.join(model_dir, "brits_demo")
        os.makedirs(brits_model_dir, exist_ok=True)
        
        print("   → Running hyperparameter optimization (this may take a while)...")
        best_params = brits_imputer.optimize_hyperparameters(
            sample_df,
            brits_model_dir,
            n_trials=10,  # Small number for demo
            timeout=300   # 5 minutes timeout
        )
        
        if best_params:
            print(f"   ✓ Optimization completed!")
            print(f"   ✓ Best parameters: {best_params}")
        else:
            print("   ✗ Optimization failed")
            
    except Exception as e:
        print(f"   ✗ Error during BRITS optimization: {e}")
    
    # Demonstrate configuration management
    print("\n4. Demonstrating configuration management...")
    try:
        # Check if config files exist
        saits_config_path = "../../params/saits.yaml"
        if os.path.exists(saits_config_path):
            config_manager = load_model_config("saits", "../../params")
            
            print(f"   ✓ Loaded SAITS configuration")
            print(f"   ✓ Current parameters: {list(config_manager.get_params().keys())}")
            print(f"   ✓ Is optimized: {config_manager.is_optimized()}")
            print(f"   ✓ Optimization history entries: {len(config_manager.get_optimization_history())}")
        else:
            print(f"   ! SAITS configuration not found: {saits_config_path}")
    
    except Exception as e:
        print(f"   ✗ Error managing configuration: {e}")
    
    print("\n5. Summary")
    print("   ✓ Demonstrated PyPOTS imputation framework")
    print("   ✓ Showed hyperparameter optimization with Optuna")
    print("   ✓ Illustrated configuration management")
    print("   ✓ Framework ready for production use")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def show_optimization_tips():
    """Show tips for effective hyperparameter optimization"""
    
    print("\n" + "="*50)
    print("HYPERPARAMETER OPTIMIZATION TIPS")
    print("="*50)
    
    tips = [
        "1. Start with a small number of trials (10-20) to test the setup",
        "2. Use longer timeouts for production optimization (1-6 hours)",
        "3. More trials generally lead to better results (50-200 trials)",
        "4. Monitor GPU/CPU usage during optimization",
        "5. Save optimization reports for analysis",
        "6. Use the same dataset for optimization and final training",
        "7. Consider early stopping with pruning for faster optimization",
        "8. Backup configuration files before running optimization"
    ]
    
    for tip in tips:
        print(f"   {tip}")
    
    print("\nRecommended settings:")
    print("   • Quick test: 10 trials, 300s timeout")
    print("   • Development: 50 trials, 1800s timeout") 
    print("   • Production: 100-200 trials, 3600-7200s timeout")
    print("="*50)


if __name__ == "__main__":
    try:
        demonstrate_optimization()
        show_optimization_tips()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        sys.exit(1)