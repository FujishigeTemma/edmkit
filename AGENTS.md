# AGENTS.md

## Project Overview

`edmkit` is a Python library for Empirical Dynamic Modeling (EDM), providing tools for nonlinear time series analysis including embedding, simplex projection, S-Map, and convergent cross mapping.

## Development Commands

### Testing
```bash
# Run all tests
uv run pytest tests/

# Run a specific test file
uv run pytest tests/test_simplex_projection.py

# Run tests with verbose output
uv run pytest -v tests/
```

### Type Checking
```bash
# Type check
uv run ty check .
```

### Linting and Formatting
```bash
# Check code style and linting issues
uv run ruff check .

# Auto-format code
uv run ruff format .

# Check specific file
uv run ruff check src/edmkit/simplex_projection.py
```

### Installation and Dependencies
```bash
# Install in development mode with uv (recommended)
uv sync --dev

# Do not use pip interface
```

## Architecture and Code Structure

### Core Modules
- **embedding.py**: Time series embedding using lagged coordinates. Key function: `lagged_embed()`
- **simplex_projection.py**: Nearest-neighbor forecasting in phase space. Key function: `simplex_projection()`
- **smap.py**: Sequential Locally Weighted Global Linear Maps. Key function: `smap()`
- **ccm.py**: Convergent Cross Mapping for causality detection. Key function: `ccm()`
- **util.py**: Utility functions including distance calculations, autocorrelation, DTW

### Key Design Patterns
1. **Functional API**: Most functions are pure with NumPy array inputs/outputs
2. **GPU Acceleration**: Uses tinygrad tensors internally for performance
3. **Input Validation**: Extensive assertions with informative error messages
4. **Vectorized Operations**: Leverages NumPy/tinygrad broadcasting for efficiency
5. **Minimal Dependency**: Use minimal dependencies and all of usage should be justified

### Testing Approach
- Tests compare outputs against pyEDM reference implementation
- Use synthetic data from generators (Lorenz, Mackey-Glass, etc.) as fixture
- Test files mirror source structure (e.g., `test_simplex_projection.py` for `simplex_projection.py`)

### Important Implementation Details
1. **Distance Calculations**: The library uses custom distance functions that handle NaN values specially
2. **Platform Considerations**: Special handling for macOS non-ARM systems regarding Metal backend
3. **API Stability**: The library is under active development - expect API changes
4. **Tensor Conversion**: Functions convert between NumPy arrays and tinygrad tensors internally
