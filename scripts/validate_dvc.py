"""
DVC integration for data validation

This script provides integration between the data validation framework
and DVC pipeline stages.
"""

import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from validation.data_validator import DataValidator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running with uv in the project environment:")
    print("   uv run python scripts/validate_dvc.py validate_pipeline")
    sys.exit(1)


def validate_stage_data(stage_name: str, config_path: str | None = None) -> bool:
    """
    Validate data for a specific DVC pipeline stage
    
    Args:
        stage_name: Name of the DVC stage (e.g., 'preprocess', 'prepare', 'impute')
        config_path: Optional path to validation configuration
        
    Returns:
        True if all data is valid, False otherwise
    """
    # Map stage names to data directories
    stage_mapping = {
        'preprocess': ('data/02_preproc', 'processed_*.csv'),
        'prepare': ('data/03_prepared', 'prepared_*.csv'),
        'impute': ('data/04_working_files', 'ts_feature_*.csv'),
        'evaluate': ('data/05_imputed', 'ts_feature_*.csv')
    }
    
    if stage_name not in stage_mapping:
        print(f"❌ Unknown stage: {stage_name}")
        print(f"Available stages: {', '.join(stage_mapping.keys())}")
        return False
    
    directory, pattern = stage_mapping[stage_name]
    
    print(f"🔍 Validating {stage_name} stage data...")
    print(f"   Directory: {directory}")
    print(f"   Pattern: {pattern}")
    
    validator = DataValidator(config_path)
    results = validator.validate_directory(directory, pattern)
    
    # Check results
    valid_count = sum(1 for r in results.values() if r.is_valid)
    total_count = len(results)
    
    if total_count == 0:
        print(f"⚠️  No files found in {directory} matching {pattern}")
        return False
    
    print(f"📊 Results: {valid_count}/{total_count} files valid")
    
    # Show any errors
    errors_found = False
    for file_path, result in results.items():
        if not result.is_valid:
            errors_found = True
            print(f"❌ {Path(file_path).name}:")
            for error in result.errors:
                print(f"   • {error}")
    
    if not errors_found:
        print("✅ All files passed validation")
    
    return valid_count == total_count


def pre_stage_validation(stage_name: str) -> bool:
    """
    Run validation before a DVC stage executes
    
    Args:
        stage_name: Name of the DVC stage
        
    Returns:
        True if validation passes, False if stage should not run
    """
    # Define input dependencies for each stage
    dependencies = {
        'prepare': [('data/02_preproc', 'processed_*.csv')],
        'impute': [('data/03_prepared', 'prepared_*.csv')],
        'evaluate': [('data/05_imputed', 'ts_feature_*.csv')]
    }
    
    if stage_name not in dependencies:
        print(f"ℹ️  No pre-validation configured for stage: {stage_name}")
        return True
    
    print(f"🔍 Pre-validation for {stage_name} stage...")
    
    validator = DataValidator()
    all_valid = True
    
    for directory, pattern in dependencies[stage_name]:
        print(f"   Checking: {directory}/{pattern}")
        results = validator.validate_directory(directory, pattern)
        
        if not results:
            print(f"❌ No files found in {directory}")
            all_valid = False
            continue
        
        for file_path, result in results.items():
            if not result.is_valid:
                all_valid = False
                print(f"❌ Invalid input: {Path(file_path).name}")
                for error in result.errors[:3]:  # Show first 3 errors
                    print(f"   • {error}")
                if len(result.errors) > 3:
                    print(f"   • ... and {len(result.errors) - 3} more errors")
    
    if all_valid:
        print("✅ Pre-validation passed")
    else:
        print("❌ Pre-validation failed - stage should not run")
    
    return all_valid


def create_dvc_validation_stage():
    """Create a DVC stage for data validation"""
    stage_config = {
        'validate_data': {
            'cmd': 'uv run python scripts/validate_dvc.py validate_pipeline',
            'deps': [
                'data/02_preproc',
                'data/03_prepared', 
                'data/04_working_files'
            ],
            'outs': [
                'validation_report.txt'
            ]
        }
    }
    
    print("DVC stage configuration for validation:")
    print(yaml.dump(stage_config, default_flow_style=False))
    
    # Check if dvc.yaml exists and append
    dvc_path = Path("dvc.yaml")
    if dvc_path.exists():
        with open(dvc_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        
        if 'stages' not in existing_config:
            existing_config['stages'] = {}
        
        existing_config['stages'].update(stage_config)
        
        with open(dvc_path, 'w') as f:
            yaml.dump(existing_config, f, default_flow_style=False)
        
        print(f"✅ Added validation stage to {dvc_path}")
    else:
        print("❌ dvc.yaml not found - cannot add validation stage")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="DVC Data Validation Integration")
    parser.add_argument("command", choices=[
        "validate_stage", 
        "pre_validate", 
        "validate_pipeline",
        "create_stage"
    ])
    parser.add_argument("--stage", help="Stage name for validation")
    parser.add_argument("--config", help="Validation configuration file")
    parser.add_argument("--output", help="Output report file")
    
    args = parser.parse_args()
    
    if args.command == "validate_stage":
        if not args.stage:
            print("❌ Stage name required for validate_stage")
            sys.exit(1)
        success = validate_stage_data(args.stage, args.config)
        sys.exit(0 if success else 1)
    
    elif args.command == "pre_validate":
        if not args.stage:
            print("❌ Stage name required for pre_validate")
            sys.exit(1)
        success = pre_stage_validation(args.stage)
        sys.exit(0 if success else 1)
    
    elif args.command == "validate_pipeline":
        # Full pipeline validation
        validator = DataValidator(args.config)
        results = validator.validate_pipeline_data()
        
        # Flatten results
        all_results = {}
        for stage, stage_results in results.items():
            all_results.update(stage_results)
        
        # Generate report
        output_path = args.output or "validation_report.txt"
        report = validator.generate_validation_report(all_results, output_path)
        
        if not args.output:  # Print to console if no output file
            print(report)
        
        # Exit with error code if any validation failed
        all_valid = all(r.is_valid for r in all_results.values())
        sys.exit(0 if all_valid else 1)
    
    elif args.command == "create_stage":
        create_dvc_validation_stage()
    
    else:
        parser.print_help()