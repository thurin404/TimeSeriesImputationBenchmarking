#!/usr/bin/env python3
"""
Command-line data validation script for time series imputation pipeline
"""

import argparse
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from validation.data_validator import DataValidator, create_validation_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running with uv in the project environment:")
    print("   uv run python validate_data.py --pipeline")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Validate data for time series imputation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate single file
  python validate_data.py --file data/03_prepared/prepared_org.csv
  
  # Validate entire pipeline
  python validate_data.py --pipeline
  
  # Validate directory with custom config
  python validate_data.py --directory data/04_working_files --config config/validation.yaml
  
  # Generate validation report
  python validate_data.py --pipeline --output validation_report.txt
  
  # Create default configuration
  python validate_data.py --create-config config/validation.yaml
        """
    )
    
    parser.add_argument(
        "--file", 
        help="Validate single CSV file"
    )
    parser.add_argument(
        "--directory", 
        help="Validate all CSV files in directory"
    )
    parser.add_argument(
        "--pipeline", 
        action="store_true",
        help="Validate entire pipeline data (recommended)"
    )
    parser.add_argument(
        "--config", 
        help="Path to validation configuration file"
    )
    parser.add_argument(
        "--output", 
        help="Save validation report to file"
    )
    parser.add_argument(
        "--create-config", 
        help="Create default validation configuration file"
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern for directory validation (default: *.csv)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only show errors and warnings, not full report"
    )
    
    args = parser.parse_args()
    
    # Create configuration if requested
    if args.create_config:
        create_validation_config(args.create_config)
        print(f"✅ Created validation configuration: {args.create_config}")
        return 0
    
    # Validate input arguments
    if not any([args.file, args.directory, args.pipeline]):
        parser.print_help()
        print("\n❌ Error: Please specify --file, --directory, or --pipeline")
        return 1
    
    try:
        # Initialize validator
        validator = DataValidator(args.config)
        
        if args.config:
            print(f"📋 Using configuration: {args.config}")
        
        # Perform validation
        if args.file:
            print(f"🔍 Validating file: {args.file}")
            result = validator.validate_file(args.file)
            
            if args.quiet:
                if result.errors:
                    print("❌ ERRORS:")
                    for error in result.errors:
                        print(f"  • {error}")
                if result.warnings:
                    print("⚠️  WARNINGS:")
                    for warning in result.warnings:
                        print(f"  • {warning}")
            else:
                print(result)
                
            return 0 if result.is_valid else 1
            
        elif args.directory:
            print(f"🔍 Validating directory: {args.directory}")
            results = validator.validate_directory(args.directory, args.pattern)
            
        elif args.pipeline:
            print("🔍 Validating entire pipeline...")
            results = validator.validate_pipeline_data()
            # Flatten results
            all_results = {}
            for stage, stage_results in results.items():
                for file_path, result in stage_results.items():
                    all_results[f"{stage}/{Path(file_path).name}"] = result
            results = all_results
        
        # Generate and display report
        if not args.file:  # Only for directory and pipeline validation
            if args.quiet:
                # Show summary only
                valid_count = sum(1 for r in results.values() if r.is_valid)
                invalid_count = len(results) - valid_count
                total_errors = sum(len(r.errors) for r in results.values())
                total_warnings = sum(len(r.warnings) for r in results.values())
                
                print("\n📊 SUMMARY:")
                print(f"  ✅ Valid files: {valid_count}")
                print(f"  ❌ Invalid files: {invalid_count}")
                print(f"  🚨 Total errors: {total_errors}")
                print(f"  ⚠️  Total warnings: {total_warnings}")
                
                # Show errors for invalid files
                if invalid_count > 0:
                    print("\n❌ FILES WITH ERRORS:")
                    for file_path, result in results.items():
                        if not result.is_valid:
                            print(f"  {file_path}:")
                            for error in result.errors:
                                print(f"    • {error}")
            else:
                # Full report
                report = validator.generate_validation_report(results, args.output)
                print(report)
            
            # Return appropriate exit code
            all_valid = all(r.is_valid for r in results.values())
            return 0 if all_valid else 1
    
    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())