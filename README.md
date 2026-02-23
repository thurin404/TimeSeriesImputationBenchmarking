# Time Series Imputation Benchmarking

## Purpose

This project benchmarks multiple time series imputation algorithms on real-world multivariate weather data. The key innovation is using **real missing data patterns** from operational weather stations while preserving **ground truth** for accurate evaluation.

### Benchmarking Methodology

1. **Real Missing Patterns** (`/data/01_raw/org`): Weather station data with authentic missing value patterns from sensor failures, maintenance, etc.
2. **Reference Data** (`/data/01_raw/ref`): Nearly complete weather data with similar dimensions (same stations/features)
3. **Pattern Transfer**: Apply the real missing patterns from `org` to `ref` data, creating a benchmark dataset that:
   - Has realistic missing data patterns (not synthetic MCAR/MAR)
   - Preserves complete ground truth for evaluation
   - Allows accurate measurement of imputation quality

### Experimental Variables

The benchmark tests three preprocessing configurations to understand their impact on imputation performance:

| Variable | Options | Purpose |
|----------|---------|---------|
| **Resample** | `1h` (hourly) → `1d` (daily) | Does temporal aggregation affect imputation quality? |
| **Scaling** | `true` / `false` | Do different feature value ranges require normalization? |
| **Partition By** | `Wide` / `Station` / `Feature` | How does data grouping affect multivariate imputation? |

**Partition strategies:**
- **Wide**: All features from all stations in a single dataframe (full multivariate context)
- **Station**: Group by weather station (spatial separation)
- **Feature**: Group by measured variable (e.g., all temperature readings together)

## Features

### 🤖 16 Imputation Methods Implemented
- **Classical Methods** (6): Mean, Forward Fill, Interpolation, KNN, Iterative, XGBoost
- **Deep Learning** (10): SAITS, BRITS, ImputeFormer, TimesNet, TimeMixer, MOMENT, TSLANet, FreTS, GPVAE, TEFN

**Note**: Time-LLM and MissNet are not currently functional due to resource constraints and implementation issues.

### ⚡ Optimization & Performance
- **Hyperparameter Optimization**: Optuna with TPE sampler and Hyperband pruning
- **Memory Management**: Automatic low memory mode for GPU constraints
- **Environment Variables**: Complete override system for flexible configuration
- **Test Mode**: Quick validation with `TEST_RUN=1` environment variable

### 📊 Data Validation
- **Comprehensive Validation**: File structure, data quality, completeness checks
- **Pipeline Integration**: DVC stage validation and pre-execution checks
- **Automated Reporting**: Detailed validation reports with error analysis
- **Configurable Rules**: Customizable validation thresholds and parameters

### 🔧 Development Tools
- **DVC Pipeline**: Reproducible ML workflow management
- **UV Package Management**: Fast Python package and environment management
- **Comprehensive Documentation**: Usage guides, model documentation, troubleshooting

## Workflow

### Pipeline Overview

```
Raw Data → Preprocessing → Pattern Transfer → Imputation → Evaluation
   ↓             ↓              ↓                 ↓            ↓
  org/         resample      apply org        multiple     metrics vs
  ref/         scale?       patterns to       methods      ground truth
              partition     ref (ground                    
                            truth preserved)
```

### Detailed Steps

1. **Import Data** (`data/01_raw/`)
   - `org/`: Real weather data with authentic missing patterns
   - `ref/`: Nearly complete reference data
   
2. **Preprocessing** (`data/02_preproc/`)
   - Optional resampling: `1h` → `1d` (configured in `params.yaml`)
   - Data cleaning and standardization
   
3. **Preparation** (`data/03_prepared/`)
   - Extract missing patterns from `org` data
   - Apply those patterns to `ref` data
   - **Ground truth preserved**: We know the true values behind the missing entries
   
4. **Working Files** (`data/04_working_files/`)
   - Optional scaling normalization
   - Partition by: Wide/Station/Feature
   - Generate model-ready datasets

5. **Imputation** (`data/05_imputed/`)
   - Apply 16 imputation algorithms
   - Each method fills the missing values
   
6. **Evaluation** (`data/06_evaluation/`)
   - Compare imputed values against ground truth
   - Calculate accuracy metrics: MAE, RMSE, SSIM, DTW distance, Cosine similarity
   - Analyze performance across preprocessing configurations
   - Results saved to `evaluation_results.csv`

### Data Validation

- **Comprehensive checks** at each pipeline stage
- **Automated validation** before and after each transformation
- **Validation reports** with error analysis
- See [Data Validation Documentation](docs/DATA_VALIDATION.md)

## Quick Start

### Prerequisites
- Python 3.8+ with [uv](https://docs.astral.sh/uv/) package manager
- CUDA-compatible GPU (optional, for deep learning models)

### Installation & Setup
```bash
# Clone repository
git clone <repository-url>
cd timeseriesimputation

# Install dependencies with uv
uv sync

# Validate data quality
uv run python validate_data.py --pipeline

# Run pipeline with test mode (quick validation)
TEST_RUN=1 uvx dvc repro

# Run full pipeline
uvx dvc repro
```

## Documentation

- 📖 **[Usage Guide](docs/USAGE_GUIDE.md)** - Complete user manual with environment variables and model selection
- 🧪 **[Experimental Design](docs/EXPERIMENTAL_DESIGN.md)** - Deep dive into preprocessing variables and their impact on imputation
- 📊 **[Evaluation Updates](docs/EVALUATION_UPDATES.md)** - How evaluation handles scaling, metadata capture, and metric comparability
- 📚 **[Models Documentation](docs/MODELS.md)** - Detailed model parameters and optimization strategies  
- 🛠️ **[Troubleshooting Guide](docs/TROUBLESHOOTING.md)** - Common issues and debugging workflows
- ✅ **[Data Validation](docs/DATA_VALIDATION.md)** - Comprehensive data quality framework

## Key Environment Variables

| Variable | Description | Default | Use Case |
|----------|-------------|---------|----------|
| `TEST_RUN` | Enable test mode (reduced epochs/trials) | `False` | Quick validation of pipeline changes |
| `LOW_MEMORY_MODE` | Force low memory mode | `False` | GPU with limited VRAM |
| `OPTUNA_N_TRIALS` | Override optimization trials | Model-specific | Faster/deeper hyperparameter search |
| `OPTUNA_TIMEOUT` | Override optimization timeout (seconds) | Model-specific | Control optimization duration |
| `MAX_EPOCHS` | Override training epochs | Model-specific | Quick experiments vs full training |
| `BATCH_SIZE` | Override batch size | Model-specific | Memory management |

## Configuration: Preprocessing Options

Edit `params.yaml` to configure the experimental setup:

```yaml
preprocessing:
  resample: true          # Resample from 1h to daily?
  resample_freq: '1d'     # Target frequency if resampling

prepare_working_files:
  partition_by: 'Wide'    # 'Wide', 'Station', or 'Feature'
  scaling: false          # Apply normalization?
```

**Impact on benchmarking:**
- **Resample**: Tests if temporal aggregation (1h→1d) improves/degrades imputation
- **Scaling**: Tests if normalizing different value ranges (temp, humidity, pressure) helps
- **Partition**: Tests whether keeping multivariate context (Wide) beats separated univariate (Feature) or spatial grouping (Station)

## Data Pipeline Stages

| Stage | Directory | Description |
|-------|-----------|-------------|
| **Raw Data** | `01_raw/org` | Real weather data with authentic missing patterns |
| | `01_raw/ref` | Nearly complete reference data (ground truth source) |
| **Preprocessing** | `02_preproc/` | Clean and optionally resample data (1h→1d) |
| **Preparation** | `03_prepared/` | Apply `org` missing patterns to `ref` (ground truth preserved) |
| **Working Files** | `04_working_files/` | Scale and partition data for models (Wide/Station/Feature) |
| **Imputation** | `05_imputed/` | Output from 16 imputation algorithms |
| **Evaluation** | `06_evaluation/` | Metrics comparing imputed vs ground truth |

**Key insight**: By stage 3 (Preparation), we have data with realistic missing patterns AND complete ground truth, enabling accurate benchmarking of imputation quality.

## Model Categories

### Classical Methods (6 models)
- **Simple**: Mean, Forward Fill, Linear Interpolation
- **Advanced**: KNN, Iterative Imputation, XGBoost

### Deep Learning Methods (10 models)
- **Transformer-based**: SAITS, ImputeFormer, MOMENT
- **RNN-based**: BRITS, TimesNet, TimeMixer
- **Lightweight**: TSLANet, FreTS
- **Generative**: GPVAE, TEFN

## Contributing

1. **Data Validation**: Always run `uv run python validate_data.py --pipeline` before commits
2. **Testing**: Use `TEST_RUN=1` for quick validation of changes
3. **Documentation**: Update relevant docs when adding features
4. **Memory**: Test with `LOW_MEMORY_MODE=1` for GPU compatibility
