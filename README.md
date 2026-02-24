# Time Series Imputation Benchmarking

## Purpose

This project benchmarks multiple time series imputation algorithms on real-world multivariate weather data. The key innovation is using **real missing data patterns** from operational weather stations while preserving **ground truth** for accurate evaluation.

The benchmarking approach is inspired by TSI-Bench [[1]](#ref-1), with a focus on realistic missing data patterns rather than synthetic missingness assumptions [[2]](#ref-2)[[3]](#ref-3).

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
- **Classical Methods** (6): Mean, Forward Fill, Interpolation, KNN [[4]](#ref-4), Iterative [[5]](#ref-5), XGBoost [[6]](#ref-6)
- **Deep Learning** (10): SAITS [[7]](#ref-7), BRITS [[8]](#ref-8), ImputeFormer [[9]](#ref-9), TimesNet [[10]](#ref-10), TimeMixer [[11]](#ref-11), MOMENT [[12]](#ref-12), TSLANet [[13]](#ref-13), FreTS [[14]](#ref-14), GPVAE [[15]](#ref-15), TEFN

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
git clone https://github.com/thurin404/TimeSeriesImputationBenchmarking.git
cd TimeSeriesImputationBenchmarking

# Install dependencies (requires uv package manager)
uv sync

# Validate data
uv run python validate_data.py --pipeline

# Run pipeline in test mode (quick validation)
TEST_RUN=1 dvc repro

# Run full pipeline
dvc repro
```


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
- **Advanced**: KNN [[4]](#ref-4), Iterative Imputation [[5]](#ref-5), XGBoost [[6]](#ref-6)

### Deep Learning Methods (10 models)
- **Transformer-based**: SAITS [[7]](#ref-7) (self-attention), ImputeFormer [[9]](#ref-9) (transformer architecture [[16]](#ref-16)), MOMENT [[12]](#ref-12) (pre-trained model)
- **RNN-based**: BRITS [[8]](#ref-8) (bidirectional RNN [[17]](#ref-17)), TimesNet [[10]](#ref-10), TimeMixer [[11]](#ref-11)
- **Lightweight**: TSLANet [[13]](#ref-13), FreTS [[14]](#ref-14) (frequency domain)
- **Generative**: GPVAE [[15]](#ref-15) (variational autoencoder [[18]](#ref-18)), TEFN

## Contributing

1. **Data Validation**: Always run `uv run python validate_data.py --pipeline` before commits
2. **Testing**: Use `TEST_RUN=1` for quick validation of changes
3. **Documentation**: Update relevant docs when adding features
4. **Memory**: Test with `LOW_MEMORY_MODE=1` for GPU compatibility

## References

<a id="ref-1"></a>[1] Du, W., Wang, J., Qian, L., Yang, Y., Ibrahim, Z., Liu, F., Wang, Z., Liu, H., Zhao, Z., Zhou, Y., Wang, W., Ding, K., Liang, Y., Prakash, B. A., & Wen, Q. (2024). TSI-Bench: Benchmarking Time Series Imputation. arXiv:2406.12747.

<a id="ref-2"></a>[2] Rubin, D. B. (1976). Inference and missing data. *Biometrika*, 63(3), 581–592.

<a id="ref-3"></a>[3] Little, R., & Rubin, D. (2019). *Statistical Analysis with Missing Data* (3rd ed.). Wiley Series in Probability and Statistics.

<a id="ref-4"></a>[4] Troyanskaya, O., et al. (2001). Missing value estimation methods for DNA microarrays. *Bioinformatics*, 17(6), 520–525.

<a id="ref-5"></a>[5] van Buuren, S., & Groothuis-Oudshoorn, K. (2011). mice: Multivariate Imputation by Chained Equations in R. *Journal of Statistical Software*, 45(3), 1–67.

<a id="ref-6"></a>[6] Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. In *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794).

<a id="ref-7"></a>[7] Du, W., Côté, D., & Liu, Y. (2023). SAITS: Self-attention-based imputation for time series. *Expert Systems with Applications*, 219, 119619.

<a id="ref-8"></a>[8] Cao, W., Wang, D., Li, J., Zhou, H., Li, L., & Li, Y. (2018). BRITS: Bidirectional Recurrent Imputation for Time Series. In *Advances in Neural Information Processing Systems* (Vol. 31).

<a id="ref-9"></a>[9] Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2024). A Time Series is Worth 64 Words: Long-term Forecasting with Transformers. In *International Conference on Learning Representations*.

<a id="ref-10"></a>[10] Wu, H., Hu, T., Liu, Y., Zhou, H., Wang, J., & Long, M. (2022). TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis. In *International Conference on Learning Representations*.

<a id="ref-11"></a>[11] Wang, S., Wu, H., Shi, X., Hu, T., Luo, H., Ma, L., Zhang, J. Y., & Zhou, J. (2025). TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting. In *International Conference on Learning Representations*.

<a id="ref-12"></a>[12] Goswami, M., Szafer, K., Choudhry, A., Cai, Y., Li, S., & Dubrawski, A. (2024). MOMENT: A Family of Open Time-series Foundation Models. In *International Conference on Machine Learning*.

<a id="ref-13"></a>[13] Eldele, E., Ragab, M., Chen, Z., Wu, M., Kwoh, C. K., Li, X., & Guan, C. (2024). TSLANet: Rethinking Transformers for Time Series Representation Learning. In *International Conference on Machine Learning*.

<a id="ref-14"></a>[14] Yi, K., Zhang, Q., Fan, W., Wang, S., Wang, P., He, H., An, N., Lian, D., Cao, L., & Niu, Z. (2023). Frequency-domain MLPs are More Effective Learners in Time Series Forecasting. In *Advances in Neural Information Processing Systems* (Vol. 36).

<a id="ref-15"></a>[15] Fortuin, V., Baranchuk, D., Rätsch, G., & Mandt, S. (2020). GP-VAE: Deep Probabilistic Time Series Imputation. In *International Conference on Artificial Intelligence and Statistics* (pp. 1651–1661).

<a id="ref-16"></a>[16] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). Attention Is All You Need. In *Advances in Neural Information Processing Systems* (Vol. 30).

<a id="ref-17"></a>[17] Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.

<a id="ref-18"></a>[18] Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. In *International Conference on Learning Representations*.
