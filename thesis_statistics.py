'''
Contains functions that releate to statistics used in this Thesis.

Functions:

1. :py:func:`bootstrap` -- Given the true labels, predicted labels of *m*
   models, as well as the metric for evaluation will bootstrap *n* times over
   the predictions and true label evaluating each time based on the metrics.
   Returns all evaluations as an array of shape = [n, m].
2. :py:func:`bootstrap_one_t_test` -- Creates a DataFrame of one tailed
   P-values for each model given a matrix of metric evaluations for each model.
   DataFrame shape = [n_models, n_models]
3. :py:func:`confidence_range` -- Given the output of :py:func:`bootstrap`
   will return the confidence range based on a P-Value level and tail for 
   each models
4. :py:func:`find_k_estimator` -- Given a list of p-values returns the number
   of those p-values that are significant at the level of alpha according to
   either the Bonferroni or Fisher method.
'''
from typing import Callable, Tuple, List

import numpy as np
import pandas as pd
from scipy import stats
import sklearn


def bootstrap(true: np.ndarray, predictions: np.ndarray,
              metric: Callable[[np.ndarray, np.ndarray], float],
              n: int = 10000, **metric_kwargs) -> np.ndarray:
    '''
    Given the true labels, predicted labels of *m* models, as well as the
    metric for evaluation will bootstrap *n* times over the predictions and
    true label evaluating each time based on the metrics. Returns all
    evaluations as an array of shape = [n, m].

    :param true: True labels, shape = [n_samples]
    :param predictions: Predictions, shape = [n_samples, n_models]
    :param metric: Function that evaluates the predictions e.g.
                   :py:func:`sklearn.metrics.accuracy_score`
    :param n: Number of times to bootstrap.
    :param **metric_kwargs: Keywords to provide to the metric function argument
    :return: Returns all *n* evaluations as a matrix, shape = [n, n_models].
    '''
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(predictions.shape[0], 1)
    vector_size = true.shape[0]
    metric_scores = np.zeros((n, predictions.shape[1]))
    for index in range(n):
        random_index = np.random.choice(vector_size, vector_size,
                                        replace=True)
        true_random = true[random_index]
        predictions_random = predictions[random_index]
        for model_index in range(predictions_random.shape[1]):
            score = metric(true_random, predictions_random[:, model_index],
                           **metric_kwargs)
            metric_scores[index, model_index] = score
    return metric_scores


def bootstrap_one_t_test(bootstrap_samples: np.ndarray,
                         model_names: List[str]) -> pd.DataFrame:
    '''
    Creates a DataFrame of one tailed P-values for each model given a matrix
    of metric evaluations for each model. DataFrame shape =
    [n_models, n_models] where the models in the columns are tested if they are
    greater than the models in the rows.

    :param bootstrap_samples: Output of :py:func:`bootstrap`. A matrix of shape
                              = [n_evaluations, n_models] where an evaluation
                              is for example an accuracy score.
    :param model_names: A list of the model names in the same order as they
                        appear in the bootstrap_samples.
    :return: A DataFrame of one tailed test for each model where the index
             and columns are labelled by the model names. Shape = [n_models,
             n_models]
    '''
    num_bootstrap_evals = bootstrap_samples.shape[0]
    num_models = bootstrap_samples.shape[1]

    p_values = np.zeros((num_models, num_models))
    for model_index in range(num_models):
        model_bootstrap = bootstrap_samples[:, model_index]
        model_bootstrap = model_bootstrap.reshape(num_bootstrap_evals, 1)
        diff = model_bootstrap - bootstrap_samples
        diff = np.sort(diff, axis=0)
        is_better = diff > 0
        first_occurence = np.argmax(is_better, axis=0)
        # Needs to check that the differences are not all zeros. If they are
        # then the first occurence is equal to the num_bootstrap_evals to
        # make the p_value as high as possible.
        last_is_better = is_better[-1, :]
        actually_better_mask = (first_occurence != 0) + last_is_better
        not_better_mask = (actually_better_mask == 0)
        not_better_values = np.full(shape=num_models,
                                    fill_value=num_bootstrap_evals)
        not_better_values *= not_better_mask
        better_values = actually_better_mask * first_occurence
        first_occurence = better_values + not_better_values

        model_p_values = first_occurence / num_bootstrap_evals
        p_values[model_index] = model_p_values
    p_values = p_values.T
    p_values = pd.DataFrame(p_values, index=model_names, columns=model_names)
    return p_values


def confidence_range(data: np.ndarray, level: float,
                     tail: str = 'one') -> Tuple[float, float]:
    '''
    Given the output of :py:func:`bootstrap` will return the confidence range
    based on a P-Value level and tail for each model.

    The tail can be either `one` or `two`. The level (P-value) is 0.05 to get
    the 95% confidence range.

    :param data: The data to genertae the confidence intervals, the output of
                 :py:func:`bootstrap`
    :param level: The P-value
    :param tail: Type of Tailed test e.g. `one` or `two`. If one always assumes
                 to remove the first level percentage of data.
    :return: The interval range e.g. for a two tailed test with level = 0.05
             will return the data at 2.5% and 97.5%. For One it will return
             data at 5% and 100%.
    '''
    data = np.sort(data)
    num_to_remove = int(level * len(data))
    if tail.lower() == 'one':
        data = data[num_to_remove:]
    elif tail.lower() == 'two':
        num_to_remove = int(num_to_remove / 2)
        data = data[num_to_remove:]
        data = data[:len(data) - num_to_remove]
    else:
        raise ValueError('tail has to be either `one` or `two` '
                         f'not {tail}')
    return data[0], data[-1]


def find_k_estimator(p_values: List[float], alpha: float,
                     method: str = 'B') -> int:
    '''
    Given a list of p-values returns the number of those p-values that are
    significant at the level of alpha according to either the Bonferroni or
    Fisher method.

    This code has come from `Dror et al. 2017 paper <https://aclanthology.coli\
    .uni-saarland.de/papers/Q17-1033/q17-1033>`_.
    Code base for the paper `here <https://github.com/rtmdrr/replicability-an\
    alysis-NLP/blob/master/Replicability_Analysis.py>`_

    Fisher is used if the p-values have come from an indepedent set i.e. method
    p-values results from indepedent datasets. Bonferroni used if this
    indepedent assumption is not True.

    :param p_values: list of p-values.
    :param alpha: significance level.
    :param method: 'B' for Bonferroni or 'F' for Fisher default Bonferroni.
    :return: Number of datasets that are significant at the level of alpha for
             the p_values given.
    '''

    n = len(p_values)
    pc_vec = [1.0] * n
    k_hat = 0
    p_values = sorted(p_values, reverse=True)
    for u in range(0, n):
        if (u == 0):
            pc_vec[u] = _calc_partial_cunjunction(p_values, u + 1, method)
        else:
            pc_vec[u] = max(_calc_partial_cunjunction(p_values, u + 1, method),
                            pc_vec[u - 1])
    k_hat = len([i for i in pc_vec if i <= alpha])
    return k_hat


def _calc_partial_cunjunction(p_values: List[float], u: int,
                              method: str = 'B') -> float:
    '''
    This function calculates the partial conjunction p-value of u out of n.

    This code has come from `Dror et al. 2017 paper <https://aclanthology.coli\
    .uni-saarland.de/papers/Q17-1033/q17-1033>`_.
    Code base for the paper `here <https://github.com/rtmdrr/replicability-an\
    alysis-NLP/blob/master/Replicability_Analysis.py>`_

    :param p_values: list of p-values.
    :param u: number of hypothesized true null hypotheses.
    :param method: 'B' for Bonferroni or 'F' for Fisher default Bonferroni.
    :return: Number of datasets that are significant at the level of alpha for
             the p_values given.
    '''
    n = len(p_values)
    sorted_pvlas = p_values[0:(n - u + 1)]
    if (method == 'B'):
        p_u_n = (n - u + 1) * min(sorted_pvlas)
    elif (method == 'F'):
        sum_chi_stat = 0
        for p in sorted_pvlas:
            sum_chi_stat = sum_chi_stat - 2 * np.log(p)
        p_u_n = 1 - stats.chi2.cdf(sum_chi_stat, 2 * (n - u + 1))

    return p_u_n
