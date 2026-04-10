# FPAD Test Suite

Comprehensive test suite for the Fingerprint Presentation Attack Detection project.

## Quick Start

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test file
pytest tests/test_feature_extraction.py -v

# Run specific test
pytest tests/test_feature_extraction.py::TestComputeGabor::test_returns_tuple -v

# Run only unit tests (skip slow integration tests)
pytest tests/ -v -m "not slow"

# Run in parallel (faster)
pytest tests/ -n auto
```

## Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_preprocessing.py    # Preprocessing tests (12 tests)
├── test_feature_extraction.py  # Feature module tests (25 tests)
├── test_feature_engineering.py # Feature engineering tests (15 tests)
├── test_feature_extractor.py   # Unified extractor tests (12 tests)
├── test_train.py            # Training pipeline tests (10 tests)
├── test_integration.py      # Integration tests (8 tests)
├── test_real_data.py        # Real data tests (requires dataset)
└── test_mlflow_logging.py   # MLflow/mock API tests (15 tests)
```

## Test Categories

### Unit Tests (`-m unit`)
Test individual components in isolation:
- `test_preprocessing.py` - Foreground extraction
- `test_feature_extraction.py` - Individual feature functions (Gabor, FDA, LCS, etc.)
- `test_feature_engineering.py` - Derived feature creation

### Integration Tests (`-m integration`)
Test component interactions:
- `test_integration.py` - Full pipeline tests
- `test_mlflow_logging.py` - MLflow integration with mocking

### Real Data Tests (`-m real_data`)
Test with actual fingerprint images:
- `test_real_data.py` - Requires `data/train/live/` and `data/train/fake/`

## Coverage Report

```bash
# Generate HTML coverage report
pytest tests/ --cov=. --cov-report=html

# Open in browser
# macOS: open coverage_report/index.html
# Windows: start coverage_report/index.html
# Linux: xdg-open coverage_report/index.html
```

### Target Coverage

| Module | Target | Status |
|--------|--------|--------|
| preprocessing | 80% | ✅ |
| feature_extraction/* | 85% | ✅ |
| feature_engineering | 90% | ✅ |
| feature_extractor | 85% | ✅ |
| training | 70% | ⏳ |

## CI/CD Integration

Tests run automatically on:
- Every push to `main` branch
- Every pull request
- Weekly scheduled runs (Monday midnight)

### GitHub Actions Workflows

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push, PR | Unit tests on 3 OS × 3 Python versions |
| `cd.yml` | Manual, Weekly | Full training pipeline |
| `real_data_tests.yml` | Manual | Tests with real dataset |

## Mocking External Services

The test suite uses mocking for:
- **MLflow**: Avoid actual server calls
- **File I/O**: Use temporary directories
- **Network**: No real API calls

Example:
```python
from unittest.mock import patch

@patch('mlflow.log_metrics')
def test_logs_metrics(mock_log_metrics):
    # Test without actual MLflow connection
    ...
```

## Fixtures

Common test data is provided via fixtures in `conftest.py`:

```python
def test_something(sample_fingerprint_image, sample_mask):
    # 512x512 synthetic fingerprint
    ...

def test_engineering(sample_feature_df):
    # DataFrame with realistic feature values
    ...
```

## Troubleshooting

### "No module named 'preprocessing'"
```bash
export PYTHONPATH=$PWD
# or
pip install -e .
```

### Real data tests skipped
```bash
# Ensure dataset is in correct location
ls data/train/live/*.png
ls data/train/fake/*.png
```

### Coverage too low
```bash
# Check which lines aren't covered
pytest tests/ --cov=. --cov-report=term-missing
```

## Adding New Tests

1. Create new test file: `tests/test_your_module.py`
2. Use fixtures from `conftest.py`
3. Follow naming convention: `test_*` functions, `Test*` classes
4. Add markers if needed: `@pytest.mark.slow`

Example:
```python
import pytest

@pytest.mark.unit
def test_your_feature(sample_image_pair):
    img, mask = sample_image_pair
    result = your_function(img, mask)
    assert result is not None
```

## Test Data Policy

- **Unit tests**: Use synthetic data (fixtures in `conftest.py`)
- **Integration tests**: Use synthetic data with full pipeline
- **Real data tests**: Require actual fingerprint dataset (skipped if not available)
