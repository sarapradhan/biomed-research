# Contributing to fmri-pipeline

Thank you for your interest in contributing to fmri-pipeline! This document provides guidelines for contributing to this project.

## How to Contribute

### Reporting Issues

If you encounter a bug, unexpected behavior, or have a feature request, please open an issue on the GitHub repository. When reporting bugs, include:

- A description of the problem
- Steps to reproduce the issue
- Your operating system, Python version, and relevant package versions
- The YAML configuration file used (with sensitive paths redacted)
- Any error messages or log output

### Submitting Changes

1. Fork the repository and create a feature branch from `main`.
2. Make your changes in the feature branch.
3. Add or update tests in the `tests/` directory for any new functionality.
4. Ensure all tests pass by running `pytest` from the project root.
5. Update documentation (README, docstrings) if your changes affect usage.
6. Submit a pull request with a clear description of the changes and their motivation.

### Code Style

- Follow PEP 8 conventions.
- Use descriptive variable names and include docstrings for all public functions.
- Keep modules focused: each file in `src/fmri_pipeline/` should address a single analysis stage.

### Adding a New Analysis Module

1. Create a new Python file in `src/fmri_pipeline/`.
2. Implement the module following the pattern of existing modules (accept configuration dict, read from/write to structured output directory).
3. Add a corresponding test file in `tests/`.
4. Register the module in `pipeline.py` so it can be toggled via YAML configuration.
5. Document any new configuration keys in the README.

## Getting Help

If you have questions about the codebase or need guidance on a contribution, open a discussion on the repository or reach out via the issue tracker.

## Code of Conduct

Contributors are expected to maintain a respectful, inclusive, and harassment-free environment. Please treat all participants with courtesy and professionalism.
