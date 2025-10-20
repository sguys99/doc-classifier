# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a machine learning project template built for Flex AI projects. It uses `uv` for Python package management and includes:
- Core ML utilities in `src/flex_ml/`
- Streamlit demo application in `demo/`
- Configuration-driven architecture with YAML configs in `configs/`
- Jupyter notebook templates in `notebooks/`
- Data pipeline structure with raw/intermediate/processed data directories

## Environment Setup

### Initial Development Setup
```bash
# Development environment (includes pre-commit hooks, black, isort)
make init-dev
# OR
bash install.sh --dev

# Production environment
make init
# OR
bash install.sh
```

The project uses:
- Python 3.12.9 (pinned version)
- uv for dependency management
- Virtual environment in `.venv/`

### Demo Application
```bash
cd demo
streamlit run main.py
```

The demo uses a multi-page Streamlit app with login/logout functionality and navigation to multiple application sections.

## Code Quality & Formatting

### Pre-commit Hooks
Pre-commit is configured with:
- black (line-length: 105)
- isort (black profile)
- trailing-whitespace, end-of-file-fixer, mixed-line-ending
- requirements-txt-fixer
- add-trailing-comma

### Manual Formatting
```bash
make format  # Runs black and isort
```

**Important**: Always maintain line-length of 105 characters for black formatting.

## Architecture

### Directory Structure
```
src/flex_ml/           # Main package
  utils/               # Utility modules
    config_loader.py   # YAML config loading
    path.py           # Path constants for project structure
    settings.py       # Settings management
configs/              # YAML configuration files
  data.yaml          # Data paths configuration
  feature.yaml       # Feature engineering configs
  model.yaml         # Model configurations
  train.yaml         # Training pipeline configs
  configs.py         # Config loading for demo (uses AWS Secrets Manager)
demo/                 # Streamlit demo application
  main.py            # Main entry point with navigation
  page_utils.py      # Login/logout utilities
  home/, app1/, app2/  # Application pages
data/                 # Data directory (git-ignored)
  raw/               # Raw data
  intermediate/      # Intermediate processing results
  processed/         # Final processed datasets
notebooks/            # Jupyter notebooks
  template.ipynb     # Notebook template
```

### Configuration System

The project uses a centralized path management system (`src/flex_ml/utils/path.py`) that defines all project paths relative to the repository root:
- `REPO_ROOT`: Repository root (3 levels up from utils)
- `CONFIG_PATH`, `DATA_PATH`, `LOG_PATH`, `NOTEBOOK_PATH`, `SOURCE_PATH`
- Data subdirectories: `RAW_DATA_PATH`, `INTERMEDIATE_DATA_PATH`, `PROCESSED_DATA_PATH`
- Config file paths: `DATA_CONFIG_PATH`, `FEATURE_CONFIG_PATH`, `MODEL_CONFIG_PATH`

Use `load_config(path)` from `flex_ml.utils.config_loader` to load YAML configurations.

### Demo Application Architecture

The demo application (`demo/main.py`) uses Streamlit's multi-page navigation with:
- Session-based login state management
- Page routing based on login status
- AWS Secrets Manager integration for configuration (`configs/configs.py`)
- Uses `.env` and `.env.dev` files for environment-specific settings

**Note**: The demo config loader uses `eval()` on AWS Secrets Manager responses, which should be reviewed for security considerations.

## Dependencies

This project includes extensive ML/AI dependencies:
- ML frameworks: torch, lightning, scikit-learn, xgboost
- LLM/GenAI: langchain, langgraph, litellm, langchain-aws
- Data processing: pandas, polars, numpy, pyarrow
- Cloud services: boto3, azure-cognitiveservices-speech, google-cloud-aiplatform
- APIs: fastapi, streamlit
- Observability: opentelemetry instrumentation, prometheus-client
- Optimization: optuna
- Visualization: matplotlib, seaborn, shap

When adding new dependencies, use `uv add <package>` and ensure they're added to `pyproject.toml`.

## Development Workflow

1. Always activate the virtual environment: `source .venv/bin/activate`
2. Pre-commit hooks run automatically on commit (if using `--dev` setup)
3. Configuration changes go in YAML files under `configs/`
4. New utilities should be added to `src/flex_ml/utils/`
5. Data files are git-ignored; only code and configs are versioned
