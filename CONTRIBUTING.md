# Contributing to 4D Radar Diffusion

Thank you for your interest in contributing to the 4D Radar Diffusion project! This document provides guidelines for contributing to the project.

## Code of Conduct

Please be respectful and constructive in your interactions with other contributors.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- System information (OS, Python version, PyTorch version, etc.)
- Relevant logs or error messages

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Use cases for the enhancement
- Potential implementation approach (if you have ideas)

### Pull Requests

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following the coding standards below
3. **Add tests** for new functionality
4. **Update documentation** as needed
5. **Run tests** to ensure everything works
6. **Submit a pull request** with a clear description of changes

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/mlkh21/4D-Radar-Diffusion.git
cd 4D-Radar-Diffusion
```

2. Install dependencies:
```bash
pip install -r requirements.txt
cd diffusion_consistency_radar
pip install -e .
```

3. Run tests:
```bash
python -m pytest tests/
```

## Coding Standards

### Python Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 100 characters
- Use descriptive variable names

### Code Quality

- **Type Hints**: Add type hints to all function signatures
- **Docstrings**: Use docstrings for all public functions and classes
- **Comments**: Add comments for complex logic
- **Error Handling**: Handle exceptions appropriately with informative messages
- **Logging**: Use the `logging` module instead of `print()` statements

### Example

```python
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

def process_voxel_data(
    data: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, int]:
    """
    Process voxel data by applying a threshold.
    
    Args:
        data: Input voxel data of shape (H, W, Z, C)
        threshold: Threshold value for filtering (default: 0.5)
    
    Returns:
        Tuple of (processed_data, num_filtered)
            processed_data: Filtered voxel data
            num_filtered: Number of voxels filtered out
    
    Raises:
        ValueError: If data has wrong shape
    """
    if len(data.shape) != 4:
        raise ValueError(f"Expected 4D data, got {len(data.shape)}D")
    
    logger.info(f"Processing voxel data with threshold {threshold}")
    
    # Your code here
    ...
```

### Testing

- Write unit tests for new functionality
- Aim for >80% code coverage
- Test edge cases and error conditions
- Use descriptive test names

Example test:
```python
def test_process_voxel_data_with_valid_input():
    """Test that process_voxel_data works with valid input."""
    data = np.random.randn(32, 32, 16, 4)
    result, count = process_voxel_data(data, threshold=0.5)
    assert result.shape == data.shape
    assert count >= 0
```

### Documentation

- Update README.md for major changes
- Add docstrings to all public APIs
- Update configuration templates if needed
- Add examples for new features

### Git Commit Messages

- Use clear, descriptive commit messages
- Start with a verb in present tense (e.g., "Add", "Fix", "Update")
- Keep the first line under 72 characters
- Add detailed description if needed

Good examples:
```
Add type hints to dataset loader

Fix memory leak in voxel processing
- Release tensors after processing
- Add garbage collection calls

Update README with installation instructions
```

## Project Structure

When adding new files, follow this structure:

```
4D-Radar-Diffusion/
├── diffusion_consistency_radar/   # Main package
│   ├── cm/                         # Core modules
│   ├── scripts/                    # Training/inference scripts
│   └── launch/                     # Launch scripts
├── examples/                       # Usage examples
├── tests/                          # Unit tests
├── docs/                           # Additional documentation
└── README.md                       # Main documentation
```

## Review Process

1. All pull requests require review before merging
2. Address reviewer comments promptly
3. Keep pull requests focused and reasonably sized
4. Be open to feedback and suggestions

## Questions?

If you have questions about contributing, please open an issue or reach out to the maintainers.

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing! 🎉
