import numbers

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils._param_validation import HasMethods, Interval
from numpy.linalg import norm
from numpy import ceil
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing, check_random_state


class TOS(BaseOverSampler):
    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "k_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "len_limit": ["boolean"],
    }

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=None,
        len_limit=True,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.len_limit = len_limit

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0, y=None
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        # skip the first nearest neighbour, it is selected as T2
        cols = np.mod(samples_indices, nn_num.shape[1] - 1) + 1

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps, y_type, y)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(
        self, X, nn_data, nn_num, rows, cols, steps, y_type=None, y=None
    ):
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        y_type : str, int or None, default=None
            Class label of the current target classes for which we want to generate
            samples.

        y : ndarray of shape (n_samples_all,), default=None
            The true target associated with `nn_data`. Used by Borderline SMOTE-2 to
            weight the distances in the sample generation process.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        t1 = X[rows]
        t2 = nn_data[nn_num[rows, 0]]
        t3 = nn_data[nn_num[rows, cols]]

        diffs = t3 - t2

        # Limit the length of the diff vector to the length of the t1-t2
        len_lim = min(1, norm(t2 - t1) / norm(diffs)) if self.len_limit else 1
        steps = steps * len_lim

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = t1 + steps.multiply(diffs)
        else:
            X_new = t1 + steps * diffs

        return X_new.astype(X.dtype)

    def _fit_resample(self, X, y):
        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            # Critical difference with SMOTE: != instead of ==
            target_class_indices = np.flatnonzero(y == class_sample)
            other_class_indices = np.flatnonzero(y != class_sample)
            X_class = _safe_indexing(X, target_class_indices)
            other_class = _safe_indexing(X, other_class_indices)

            # use rate of target class to other class as number of nearest neighbors
            rate = ceil(other_class.shape[0] / (y.shape[0] - other_class.shape[0]))
            rate = self.k_neighbors if self.k_neighbors else max(rate, 5)
            nn_k_ = NearestNeighbors(n_neighbors=rate)

            nn_k_.fit(other_class)
            nns = nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class,
                y.dtype,
                class_sample,
                other_class,
                nns,
                n_samples,
                1.0,
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled
