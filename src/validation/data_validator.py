"""
Data Validation Framework for Time Series Imputation

This module provides comprehensive data validation for time series data
to ensure data quality and prevent pipeline failures.
"""

import pandas as pd
import numpy as np
import yaml
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidationResult:
    """Results of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]
    
    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = f"Data Validation: {status}\n"
        
        if self.errors:
            result += f"\n🚨 ERRORS ({len(self.errors)}):\n"
            for i, error in enumerate(self.errors, 1):
                result += f"  {i}. {error}\n"
        
        if self.warnings:
            result += f"\n⚠️  WARNINGS ({len(self.warnings)}):\n"
            for i, warning in enumerate(self.warnings, 1):
                result += f"  {i}. {warning}\n"
        
        if self.summary:
            result += "\n📊 SUMMARY:\n"
            for key, value in self.summary.items():
                result += f"  • {key}: {value}\n"
        
        return result


class DataValidator:
    """Comprehensive data validation for time series imputation"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize validator with optional configuration
        
        Args:
            config_path: Path to validation configuration file
        """
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration"""
        default_config = {
            "max_missing_percentage": 95.0,
            "min_valid_samples": 10,
            "max_file_size_mb": 500,
            "required_numeric_ratio": 0.8,
            "outlier_std_threshold": 5.0,
            "min_features": 1,
            "max_features": 100,
            "timestamp_formats": [
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d",
                "%m/%d/%Y %H:%M:%S",
                "%m/%d/%Y"
            ]
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
        
        return default_config
    
    def validate_file(self, file_path: str) -> ValidationResult:
        """
        Validate a single CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            ValidationResult with validation details
        """
        errors = []
        warnings = []
        summary = {}
        
        try:
            # File existence and accessibility
            if not Path(file_path).exists():
                errors.append(f"File does not exist: {file_path}")
                return ValidationResult(False, errors, warnings, summary)
            
            # File size check
            file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
            summary["file_size_mb"] = round(file_size_mb, 2)
            
            if file_size_mb > self.config["max_file_size_mb"]:
                warnings.append(f"Large file size: {file_size_mb:.1f}MB (threshold: {self.config['max_file_size_mb']}MB)")
            
            # Load and validate DataFrame
            try:
                df = pd.read_csv(file_path, index_col=0)
            except Exception as e:
                errors.append(f"Failed to read CSV file: {str(e)}")
                return ValidationResult(False, errors, warnings, summary)
            
            # Basic DataFrame validation
            validation_result = self._validate_dataframe(df, file_path)
            errors.extend(validation_result.errors)
            warnings.extend(validation_result.warnings)
            summary.update(validation_result.summary)
            
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def _validate_dataframe(self, df: pd.DataFrame, file_path: str) -> ValidationResult:
        """Validate DataFrame content"""
        errors = []
        warnings = []
        summary = {}
        
        # Shape validation
        n_rows, n_cols = df.shape
        summary["shape"] = f"{n_rows} x {n_cols}"
        summary["n_features"] = n_cols
        
        if n_rows < self.config["min_valid_samples"]:
            errors.append(f"Too few samples: {n_rows} (minimum: {self.config['min_valid_samples']})")
        
        if n_cols < self.config["min_features"]:
            errors.append(f"Too few features: {n_cols} (minimum: {self.config['min_features']})")
        
        if n_cols > self.config["max_features"]:
            warnings.append(f"Many features: {n_cols} (may impact performance)")
        
        # Index validation
        self._validate_index(df, errors, warnings, summary)
        
        # Column validation
        self._validate_columns(df, errors, warnings, summary)
        
        # Data type validation
        self._validate_data_types(df, errors, warnings, summary)
        
        # Missing data validation
        self._validate_missing_data(df, errors, warnings, summary)
        
        # Data quality validation
        self._validate_data_quality(df, errors, warnings, summary)
        
        is_valid = len(errors) == 0
        return ValidationResult(is_valid, errors, warnings, summary)
    
    def _validate_index(self, df: pd.DataFrame, errors: List[str], 
                       warnings: List[str], summary: Dict[str, Any]):
        """Validate DataFrame index"""
        index_info = {
            "index_name": df.index.name,
            "index_type": str(type(df.index).__name__),
            "index_duplicates": df.index.duplicated().sum()
        }
        summary.update(index_info)
        
        # Check for duplicate indices
        if index_info["index_duplicates"] > 0:
            warnings.append(f"Duplicate index values: {index_info['index_duplicates']}")
        
        # Try to parse index as datetime if it looks like timestamps
        if df.index.name and 'time' in str(df.index.name).lower():
            try:
                pd.to_datetime(df.index)
                summary["timestamp_parseable"] = True
            except Exception:
                warnings.append("Index appears to be timestamps but cannot be parsed as datetime")
                summary["timestamp_parseable"] = False
    
    def _validate_columns(self, df: pd.DataFrame, errors: List[str], 
                         warnings: List[str], summary: Dict[str, Any]):
        """Validate DataFrame columns"""
        # Column names
        column_issues = []
        
        for col in df.columns:
            if pd.isna(col) or str(col).strip() == '':
                column_issues.append("Empty or NaN column name")
            elif str(col).startswith('Unnamed:'):
                column_issues.append(f"Unnamed column: {col}")
        
        if column_issues:
            warnings.extend(column_issues)
        
        # Duplicate column names
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"Duplicate column names: {duplicate_cols}")
        
        summary["column_names"] = df.columns.tolist()
        summary["n_unnamed_columns"] = sum(1 for col in df.columns if str(col).startswith('Unnamed:'))
    
    def _validate_data_types(self, df: pd.DataFrame, errors: List[str], 
                            warnings: List[str], summary: Dict[str, Any]):
        """Validate data types"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        object_cols = df.select_dtypes(include=['object']).columns
        
        numeric_ratio = len(numeric_cols) / len(df.columns) if len(df.columns) > 0 else 0
        
        summary.update({
            "numeric_columns": len(numeric_cols),
            "object_columns": len(object_cols),
            "numeric_ratio": round(numeric_ratio, 3)
        })
        
        if numeric_ratio < self.config["required_numeric_ratio"]:
            warnings.append(f"Low numeric ratio: {numeric_ratio:.1%} (expected: {self.config['required_numeric_ratio']:.1%})")
        
        # Check for columns that should be numeric but aren't
        potentially_numeric = []
        for col in object_cols:
            try:
                pd.to_numeric(df[col], errors='coerce')
                potentially_numeric.append(col)
            except Exception:
                pass
        
        if potentially_numeric:
            warnings.append(f"Columns that might be convertible to numeric: {potentially_numeric}")
    
    def _validate_missing_data(self, df: pd.DataFrame, errors: List[str], 
                              warnings: List[str], summary: Dict[str, Any]):
        """Validate missing data patterns"""
        # Overall missing percentage
        total_cells = df.size
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells * 100) if total_cells > 0 else 0
        
        summary.update({
            "total_missing_cells": missing_cells,
            "missing_percentage": round(missing_percentage, 2)
        })
        
        if missing_percentage > self.config["max_missing_percentage"]:
            errors.append(f"Too much missing data: {missing_percentage:.1f}% (max: {self.config['max_missing_percentage']}%)")
        elif missing_percentage > 50:
            warnings.append(f"High missing data: {missing_percentage:.1f}%")
        
        # Per-column missing analysis
        col_missing = df.isnull().sum()
        completely_missing = col_missing[col_missing == len(df)].index.tolist()
        mostly_missing = col_missing[col_missing > len(df) * 0.9].index.tolist()
        
        if completely_missing:
            errors.append(f"Completely empty columns: {completely_missing}")
        
        if mostly_missing:
            warnings.append(f"Mostly empty columns (>90% missing): {mostly_missing}")
        
        summary.update({
            "completely_missing_columns": completely_missing,
            "mostly_missing_columns": mostly_missing,
            "max_column_missing_pct": round((col_missing.max() / len(df) * 100), 2) if len(df) > 0 else 0
        })
    
    def _validate_data_quality(self, df: pd.DataFrame, errors: List[str], 
                              warnings: List[str], summary: Dict[str, Any]):
        """Validate data quality issues"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            warnings.append("No numeric columns found for quality analysis")
            return
        
        # Infinite values
        inf_counts = {}
        for col in numeric_df.columns:
            inf_count = np.isinf(numeric_df[col]).sum()
            if inf_count > 0:
                inf_counts[col] = inf_count
        
        if inf_counts:
            errors.append(f"Infinite values found: {inf_counts}")
        
        # Extreme outliers (beyond 5 standard deviations)
        outlier_info = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                mean_val = col_data.mean()
                std_val = col_data.std()
                if std_val > 0:
                    outliers = np.abs(col_data - mean_val) > (self.config["outlier_std_threshold"] * std_val)
                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        outlier_info[col] = {
                            "count": outlier_count,
                            "percentage": round(outlier_count / len(col_data) * 100, 2)
                        }
        
        if outlier_info:
            warning_msg = "Potential outliers detected: "
            warning_msg += ", ".join([f"{col}: {info['count']} ({info['percentage']}%)" 
                                    for col, info in outlier_info.items()])
            warnings.append(warning_msg)
        
        # Data range validation
        range_info = {}
        for col in numeric_df.columns:
            col_data = numeric_df[col].dropna()
            if len(col_data) > 0:
                range_info[col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "range": float(col_data.max() - col_data.min())
                }
        
        summary.update({
            "outlier_analysis": outlier_info,
            "data_ranges": range_info,
            "infinite_values": inf_counts
        })
    
    def validate_directory(self, directory_path: str, 
                          file_pattern: str = "*.csv") -> Dict[str, ValidationResult]:
        """
        Validate all files in a directory
        
        Args:
            directory_path: Path to directory containing data files
            file_pattern: Glob pattern for files to validate
            
        Returns:
            Dictionary mapping file paths to validation results
        """
        results = {}
        directory = Path(directory_path)
        
        if not directory.exists():
            dummy_result = ValidationResult(
                False, 
                [f"Directory does not exist: {directory_path}"], 
                [], 
                {}
            )
            return {directory_path: dummy_result}
        
        files = list(directory.glob(file_pattern))
        
        if not files:
            dummy_result = ValidationResult(
                False, 
                [f"No files found matching pattern '{file_pattern}' in {directory_path}"], 
                [], 
                {}
            )
            return {directory_path: dummy_result}
        
        for file_path in files:
            try:
                results[str(file_path)] = self.validate_file(str(file_path))
            except Exception as e:
                results[str(file_path)] = ValidationResult(
                    False, 
                    [f"Validation failed: {str(e)}"], 
                    [], 
                    {}
                )
        
        return results
    
    def validate_pipeline_data(self, base_path: str = "data") -> Dict[str, Dict[str, ValidationResult]]:
        """
        Validate data across all pipeline stages
        
        Args:
            base_path: Base path to data directory
            
        Returns:
            Nested dictionary with stage and file validation results
        """
        stages = {
            "02_preproc": "processed_*.csv",
            "03_prepared": "prepared_*.csv", 
            "04_working_files": "ts_feature_*.csv"
        }
        
        results = {}
        
        for stage, pattern in stages.items():
            stage_path = Path(base_path) / stage
            if stage_path.exists():
                results[stage] = self.validate_directory(str(stage_path), pattern)
            else:
                results[stage] = {
                    str(stage_path): ValidationResult(
                        False,
                        [f"Stage directory does not exist: {stage_path}"],
                        [],
                        {}
                    )
                }
        
        return results
    
    def generate_validation_report(self, results: Dict[str, ValidationResult], 
                                 output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            results: Dictionary of validation results
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        report = []
        report.append("=" * 80)
        report.append("DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {pd.Timestamp.now()}")
        report.append(f"Total files validated: {len(results)}")
        
        # Summary statistics
        valid_count = sum(1 for r in results.values() if r.is_valid)
        invalid_count = len(results) - valid_count
        total_errors = sum(len(r.errors) for r in results.values())
        total_warnings = sum(len(r.warnings) for r in results.values())
        
        report.append("\n📊 SUMMARY:")
        report.append(f"  ✅ Valid files: {valid_count}")
        report.append(f"  ❌ Invalid files: {invalid_count}")
        report.append(f"  🚨 Total errors: {total_errors}")
        report.append(f"  ⚠️  Total warnings: {total_warnings}")
        
        # Detailed results
        report.append("\n📋 DETAILED RESULTS:")
        report.append("-" * 80)
        
        for file_path, result in results.items():
            report.append(f"\nFile: {file_path}")
            report.append(str(result))
        
        report_text = "\n".join(report)
        
        # Save to file if requested
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(report_text)
        
        return report_text


def create_validation_config(output_path: str = "config/validation.yaml"):
    """Create a default validation configuration file"""
    config = {
        "max_missing_percentage": 95.0,
        "min_valid_samples": 10,
        "max_file_size_mb": 500,
        "required_numeric_ratio": 0.8,
        "outlier_std_threshold": 5.0,
        "min_features": 1,
        "max_features": 100,
        "timestamp_formats": [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M:%S",
            "%m/%d/%Y"
        ]
    }
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Validation configuration created at: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validation Tool")
    parser.add_argument("--file", help="Validate single file")
    parser.add_argument("--directory", help="Validate directory")
    parser.add_argument("--pipeline", action="store_true", help="Validate entire pipeline")
    parser.add_argument("--config", help="Path to validation configuration")
    parser.add_argument("--output", help="Output report path")
    parser.add_argument("--create-config", help="Create default config file")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_validation_config(args.create_config)
        exit(0)
    
    validator = DataValidator(args.config)
    
    if args.file:
        result = validator.validate_file(args.file)
        print(result)
    elif args.directory:
        results = validator.validate_directory(args.directory)
        report = validator.generate_validation_report(results, args.output)
        print(report)
    elif args.pipeline:
        results = validator.validate_pipeline_data()
        all_results = {}
        for stage, stage_results in results.items():
            all_results.update(stage_results)
        report = validator.generate_validation_report(all_results, args.output)
        print(report)
    else:
        print("Please specify --file, --directory, or --pipeline")