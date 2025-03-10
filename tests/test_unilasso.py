import pytest
import numpy as np
import pandas as pd
from unilasso import fit_unilasso, cv_unilasso, simulate_cox_data, simulate_gaussian_data, predict, extract_cv_unilasso




def _check_result(result, p, family):
    assert hasattr(result, "coefs")
    assert hasattr(result, "intercept")
    assert hasattr(result, "_gamma")
    assert hasattr(result, "_beta")

    assert result.coefs.shape == (p,)
    assert type(result.intercept) == np.ndarray
    assert result._gamma.shape == (p,)
    assert result._beta.shape == (p,)

    if family == "cox":
        # Cox model has an additional intercept term
        assert result.intercept == 0
        assert result._gamma_intercept == 0
        assert np.all(result._beta_intercepts == 0)




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
    result = fit_unilasso(X, y, family=family, lmdas=[0.1])
    _check_result(result, 5, family)



def test_fit_gaussian():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])

    X[:, 0] = 1
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])
    assert result.coefs[0] == 0 # First coefficient should be zero
    
    _check_result(result, 5, "gaussian")



def test_fit_binomial():
    X = np.random.rand(100, 5)
    y = np.random.randint(2, size=100)
    result = fit_unilasso(X, y, family="binomial", lmdas=[0.1])
    
    _check_result(result, 5, "binomial")


def test_fit_cox():
    X, y = simulate_cox_data(n=100, p=5, seed=123)
    result = fit_unilasso(X, y, family="cox", lmdas=[0.1])
    _check_result(result, 5, "cox")

    y = pd.DataFrame(y, columns=['time', 'status'])
    result = fit_unilasso(X, y, family="cox", lmdas=[0.1])
    _check_result(result, 5, "cox")


    y = pd.DataFrame(y, columns=['time', 'trend'])
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", lmdas=[0.1])

    
def test_zero_variance():
    X = np.random.rand(100, 5)
    X[:, 0] = 1
    y = np.random.rand(100)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1])

    _check_result(result, 5, "gaussian")
    assert result._gamma[0] == 0 # First feature has zero variance
    assert result._beta[0] == 0 # First feature has zero variance

    # Model should throw error if all features have zero variance
    X = np.zeros((100, 5))
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family="cox", lmdas=[0.1])



def test_input_validation():
    X = np.random.rand(100, 5)
    y = np.random.rand(100)
    with pytest.raises(ValueError):
        fit_unilasso(X, y, family='invalid_family', lmdas=[0.1])


def _check_cv_result(result, p, family):
    num_lmdas = len(result.lmdas)
    assert hasattr(result, "cv_plot")
    assert hasattr(result, "best_idx")
    assert isinstance(result.best_idx, int)
    assert isinstance(result.lmdas, np.ndarray)
    assert result.coefs.shape == (num_lmdas, p)


def test_cv_unilasso():
    X, y = simulate_gaussian_data(n=100, p=5, seed=123)

    # Test extra arguments for cv_unilasso
    with pytest.raises(TypeError) as excinfo:
        cv_unilasso(X, y, family='gaussian', lmdas=[0.1])
        
    result = cv_unilasso(X, y, family="gaussian")
    _check_cv_result(result, 5, "gaussian")

    extracted_fit = extract_cv_unilasso(result)
    assert extracted_fit.coefs.shape == (5, )
    assert type(extracted_fit.intercept) == np.float64


def test_predict():
    X, y = simulate_gaussian_data(n=100, p=5)
    result = fit_unilasso(X, y, family="gaussian", lmdas=[0.1, 0.2])
    y_pred = predict(X, result)
    assert y_pred.shape == (100, 2)

    y_pred_lmda_1 = predict(X, result, lmda_idx=0)
    assert y_pred_lmda_1.shape == (100, )
