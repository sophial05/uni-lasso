# UniLasso: Univariate-Guided Sparse Regression

UniLasso is a Python package that implements a novel and interpretable approach to sparse regression corresponding to the preprint Univariate-Guided Sparse Regression (https://arxiv.org/abs/2501.18360).


## Features

- Implements UniLasso algorithm for various regression types (Gaussian, Binomial, Cox)
- Supports cross-validation for optimal regularization parameter selection
- Provides both fit and predict functionalities
- Includes utilities for data simulation


## Installation

You can install UniLasso by running the following script.

```bash
git clone https://github.com/sophial05/uni-lasso.git
cd uni-lasso
pip install -e .
```

## Quick Start

Here's a simple example of how to use UniLasso:
```python
import numpy as np
from unilasso import *

# Generate some example data
X, y = simulate_gaussian_data(n = 100, p = 5)

# Fit UniLasso model using default lambda path
result = fit_unilasso(X, y, family='gaussian', lmdas = None)
plot_coef_path(result)

# Run UniLasso model using CV and plot CV error curve
cv_fit = cv_unilasso(X, y, family='gaussian')
plot_cv(cv_fit)
# extract the best cv model to use for prediction
extracted_fit = extract_cv(cv_fit)

# Generate some test data
X_test, y_test = simulate_gaussian_data(n = 100, p = 5)
# predict on test data
y_pred = predict(extracted_fit, X_test)

# Print coefficients
print("UniLasso coefficients:")
print(extracted_fit.coefs)

# Predict on data
y_hat = predict(X, fit)
```

For a comprehensive examples on package usage, please refer to `examples/` subdirectory.


## License
This project is licensed under the MIT License - see the LICENSE file for details.


## Citation
If you use UniLasso in your research, please cite:
```bibtex
@article{chatterjee2025univariate,
  title={Univariate-Guided Sparse Regression},
  author={Chatterjee, Sourav and Hastie, Trevor and Tibshirani, Robert},
  journal={arXiv preprint arXiv:2501.18360},
  year={2025}
}
```

## Contact
For any questions or feedback, please contact Sophia Lu at sophialu@stanford.edu




