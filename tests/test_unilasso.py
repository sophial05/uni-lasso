import pytest
import numpy as np
import pandas as pd
from unilasso import fit_unilasso, cv_unilasso, simulate_cox_data



@pytest.mark.parametrize("family", ["gaussian", "binomial", "cox"])
def test_different_families(family):
    
    if family == "cox":
        X, y = simulate_cox_data(n=100, p=5, seed=123)
    else:
        X = np.random.randn(100, 5)
        if family == "binomial":
            y = np.random.randint(2, size=100)
        else:
            y = np.random.randn(100)
    result = fit_unilasso(X, y, family=family, regularizers=[0.1])
    assert isinstance(result, dict)



def test_fit_gaussian():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", regularizers=[0.1])
    assert isinstance(result, dict)
    assert 'coef' in result
    assert 'intercept' in result
    assert '_gamma' in result
    assert '_beta' in result
    assert result['coef'].shape == (5,)
    assert type(result['intercept']) == np.float64
    assert result['_gamma'].shape == (5,)
    assert result['_beta'].shape == (5,)



def test_fit_binomial():
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    result = fit_unilasso(X, y, family="binomial", regularizers=[0.1])
    assert isinstance(result, dict)
    assert 'coef' in result
    assert 'intercept' in result
    assert '_gamma' in result
    assert '_beta' in result
    assert result['coef'].shape == (5,)
    assert type(result['intercept']) == np.float64
    assert result['_gamma'].shape == (5,)
    assert result['_beta'].shape == (5,)


def test_fit_cox():
    X, y = simulate_cox_data(n=100, p=5, seed=123)
    result = fit_unilasso(X, y, family="cox", regularizers=[0.1])
    assert isinstance(result, dict)
    assert 'coef' in result
    assert 'intercept' in result
    assert '_gamma' in result
    assert '_beta' in result
    assert result['coef'].shape == (5,)
    assert type(result['intercept']) == np.float64
    assert result['_gamma'].shape == (5,)
    assert result['_beta'].shape == (5,)

    y = pd.DataFrame(y, columns=['time', 'status'])
    result = fit_unilasso(X, y, family="cox", regularizers=[0.1])
    assert isinstance(result, dict)
    assert 'coef' in result
    assert 'intercept' in result
    assert '_gamma' in result
    assert '_beta' in result
    assert result['coef'].shape == (5,)
    assert type(result['intercept']) == np.float64
    assert result['_gamma'].shape == (5,)
    assert result['_beta'].shape == (5,)

    y = pd.DataFrame(y, columns=['time', 'trend'])
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", regularizers=[0.1])

    
def test_zero_variance():
    X = np.random.rand(100, 5)
    X[:, 0] = 1
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", regularizers=[0.1])
    assert isinstance(result, dict)
    assert 'coef' in result
    assert 'intercept' in result
    assert '_gamma' in result
    assert '_beta' in result
    assert result['coef'].shape == (5,)
    assert type(result['intercept']) == np.float64
    assert result['_gamma'].shape == (5,)
    assert result['_beta'].shape == (5,)
    assert result['_gamma'][0] == 0
    assert result['_beta'][0] == 0

    X = np.zeros((100, 5))
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", regularizers=[0.1])



def test_input_validation():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family='invalid_family', regularizers=[0.1])


def test_cv_unilasso():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)

    # Test extra arguments for cv_unilasso
    with pytest.raises(TypeError) as excinfo:
        cv_unilasso(X, y, family='gaussian', regularizers=[0.1])
        
    result = cv_unilasso(X, y, family="gaussian")
    assert isinstance(result, dict)
    assert 'best_idx' in result
    assert isinstance(result['best_idx'], int)
    assert isinstance(result['regularizers'], np.ndarray)


