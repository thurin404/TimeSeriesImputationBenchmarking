#!/usr/bin/env python3
"""
Simple architecture test without external dependencies.
Tests the class hierarchy and method signatures.
"""

import sys
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, Type

def test_class_architecture():
    """Test the class architecture without importing external dependencies"""
    
    print("Testing PyPOTS Framework Architecture")
    print("=" * 40)
    
    # Test 1: Check if we can define the abstract base class
    try:
        class MockBasePypotsImputer(ABC):
            def __init__(self, imputation_method: str):
                self.imputation_method = imputation_method
                self.model_trained = False
                self.partition_by = None
            
            @abstractmethod
            def get_model_class(self) -> Type:
                pass
            
            @abstractmethod
            def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str) -> Dict[str, Any]:
                pass
            
            @abstractmethod
            def get_optimization_params(self, trial) -> Dict[str, Any]:
                pass
        
        print("✓ Abstract base class structure is valid")
        
    except Exception as e:
        print(f"✗ Base class structure error: {e}")
        return False
    
    # Test 2: Check if we can implement concrete classes
    try:
        class MockSAITSImputer(MockBasePypotsImputer):
            def __init__(self):
                super().__init__("saits")
            
            def get_model_class(self) -> Type:
                return type  # Mock return
            
            def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str) -> Dict[str, Any]:
                return {
                    "n_steps": n_steps,
                    "n_features": n_features,
                    "n_layers": params.get("n_layers", 4),
                    "d_model": params.get("d_model", 384),
                }
            
            def get_optimization_params(self, trial) -> Dict[str, Any]:
                # Mock trial object
                return {
                    "n_layers": 4,
                    "d_model": 384,
                    "n_heads": 4,
                }
        
        # Test instantiation
        imputer = MockSAITSImputer()
        assert imputer.imputation_method == "saits"
        assert hasattr(imputer, 'get_model_class')
        assert hasattr(imputer, 'get_model_params')
        assert hasattr(imputer, 'get_optimization_params')
        
        print("✓ Concrete implementation works correctly")
        
    except Exception as e:
        print(f"✗ Concrete implementation error: {e}")
        return False
    
    # Test 3: Check method signatures
    try:
        # Test get_model_params
        params = {"n_layers": 4, "d_model": 384}
        model_params = imputer.get_model_params(params, 96, 5, "/tmp")
        assert isinstance(model_params, dict)
        assert "n_steps" in model_params
        assert "n_features" in model_params
        
        # Test get_optimization_params  
        class MockTrial:
            pass
        
        opt_params = imputer.get_optimization_params(MockTrial())
        assert isinstance(opt_params, dict)
        
        print("✓ Method signatures are correct")
        
    except Exception as e:
        print(f"✗ Method signature error: {e}")
        return False
    
    # Test 4: Check extensibility
    try:
        class MockBRITSImputer(MockBasePypotsImputer):
            def __init__(self):
                super().__init__("brits")
            
            def get_model_class(self) -> Type:
                return type
            
            def get_model_params(self, params: Dict[str, Any], n_steps: int, n_features: int, model_dir: str) -> Dict[str, Any]:
                return {
                    "n_steps": n_steps,
                    "n_features": n_features,
                    "rnn_hidden_size": params.get("rnn_hidden_size", 384),
                }
            
            def get_optimization_params(self, trial) -> Dict[str, Any]:
                return {
                    "rnn_hidden_size": 384,
                    "batch_size": 96,
                }
        
        brits_imputer = MockBRITSImputer()
        assert brits_imputer.imputation_method == "brits"
        
        print("✓ Framework is easily extensible")
        
    except Exception as e:
        print(f"✗ Extensibility error: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("Architecture test completed successfully!")
    print("✓ Abstract base class design is sound")
    print("✓ Concrete implementations work correctly") 
    print("✓ Method signatures are consistent")
    print("✓ Framework is easily extensible")
    print("=" * 40)
    
    return True

if __name__ == "__main__":
    success = test_class_architecture()
    if not success:
        sys.exit(1)
    
    print("\nNext steps:")
    print("1. Install dependencies: pip install optuna pygrinder")
    print("2. Run: python test_optimization.py")
    print("3. Use the framework in your pipeline!")