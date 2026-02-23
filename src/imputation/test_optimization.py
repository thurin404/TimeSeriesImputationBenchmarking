#!/usr/bin/env python3
"""
Test script for hyperparameter optimization functionality.
"""

import sys
from pathlib import Path

# Add the imputation directory to the path
sys.path.insert(0, str(Path(__file__).parent))

def test_optimization_imports():
    """Test if all imports work correctly"""
    try:
        from base_pypots_imputer import BasePypotsImputer
        print("✓ Base imputer imported")
        
        from saits import SAITSImputer
        print("✓ SAITS imputer imported")
        
        from brits import BRITSImputer  
        print("✓ BRITS imputer imported")
        
        from config_manager import ModelConfigManager
        print("✓ Config manager imported")
        
        # Test imputer creation
        saits = SAITSImputer()
        brits = BRITSImputer()
        print("✓ Imputer instances created")
        
        # Test optimization method exists
        if hasattr(saits, 'optimize_hyperparameters'):
            print("✓ Optimization method available")
        else:
            print("✗ Optimization method missing")
            
        if hasattr(saits, 'get_optimization_params'):
            print("✓ Optimization params method available")
        else:
            print("✗ Optimization params method missing")
        
        print("\nAll imports successful!")
        return True
        
    except Exception as e:
        print(f"Import failed: {e}")
        return False

def test_optuna_availability():
    """Test if Optuna is available"""
    try:
        import optuna
        print("✓ Optuna is available")
        
        from optuna.samplers import TPESampler
        from optuna.pruners import HyperbandPruner
        print("✓ Optuna samplers and pruners available")
        
        return True
    except ImportError:
        print("✗ Optuna not available - optimization will be disabled")
        return False

if __name__ == "__main__":
    print("Testing PyPOTS Optimization Framework")
    print("=" * 40)
    
    imports_ok = test_optimization_imports()
    optuna_ok = test_optuna_availability()
    
    print("\n" + "=" * 40)
    if imports_ok:
        print("Framework is ready to use!")
        if optuna_ok:
            print("Hyperparameter optimization is available.")
        else:
            print("Install Optuna for hyperparameter optimization.")
    else:
        print("Framework setup incomplete.")
        sys.exit(1)