"""
This module contains ensemble binary classifiers designed for
imbalanced datasets
"""
import numpy as np
import pandas as pd
import copy


class RatioBaggingClassifier(object):
    """
    Class that defines a fit and predict_proba method for an ensemble
    classifier that combines classifiers of the same type trained at
    different positives to negatives ratio. The final score is given
    by the mean of the scores.
    """

    def __init__(self, classifier, ratios=([1, 0.8, 0.6], [1, 1, 1])):
        """

        Args:
            classifier: base classifier, sklearn-like
            ratios (tuple of int list): desired sampling ratio for the
            two classes. The first list containing the ratios for the 0
            class, the second list containing the ratios for the 1 class.
            The length of the 2 list will correspond to the number of
            classifiers combined.
        """
        self.classifier = classifier
        self.ratios = ratios
        self.models = []
        self.predictions = []

    @property
    def __name__(self):
        return "RatioBaggingClassifier"

    def fit(self, df, labels, seed=1234):
        """
        Method that trains several models, each on a sample of df that
        has imbalance ratio as described in ratios.

        Args:
            df (pd.DataFrame): df containing features
            labels (pd.Series): labels column
            seed (int): seed for the sampling

        """
        self.models = []
        label_col = "outcome"
        df = df.copy()
        df[label_col] = labels.values

        if len(self.ratios[0]) != len(self.ratios[1]):
            raise ValueError(
                "ratios has to be of the form ([1,0.8,0.6],[1,1,1])"
            )

        for i in range(len(self.ratios[0])):
            # sample
            d_ratios = {0: self.ratios[0][i], 1: self.ratios[1][i]}
            df = df.sample(frac=1).reset_index(drop=True)
            df_train_undersampled = sample_by(
                df, label_col, d_ratios, seed=seed
            )

            # prepare dataset
            Y = df_train_undersampled[label_col]
            X = df_train_undersampled.drop([label_col], axis=1)
            # fit
            print("Model %i trained" % (i + 1))
            model = copy.deepcopy(self.classifier.fit(X, Y))
            self.models.append(model)

    def predict_proba(self, X):
        """
        Method that predicts the probability score for each row in X. It
        takes the mean of the score of each classifier.

        Args:
            X (array of float): dataset containing the features

        Returns: np.array of floats containing the predictions for each
         row in X

        """
        self.predictions = []
        for model in self.models:
            self.predictions.append(model.predict_proba(X))
        return np.mean(self.predictions, axis=0)


class BaggingClassifier(object):
    """
    Class that defines a fit and predict_proba method for an ensemble
    classifier that combines classifiers of the same type trained on
    several bootstrapped samples. Positives to negatives ratio (P2Nratio)
    is the same for each classifier. The final score is given by the mean
    of the scores.
    """

    def __init__(
        self,
        classifier,
        n_estimators=10,
        d_ratios={0: 0.1, 1: 1},
        sample_feature=1,
    ):
        """

        Args:
            classifier: base classifier, sklearn-like
            n_estimators (int): number of classifier to be combined
            d_ratios (dict): sampling ratio for each class
        """
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.d_ratios = d_ratios
        self.models = []
        self.predictions = []

    @property
    def __name__(self):
        return "BaggingClassifier"

    def fit(self, df, labels, seed=1234):
        """

        Args:
          df (pd.DataFrame): df containing features
          labels (pd.Series): labels column
          seed (int): seed to initialize the sampling
        """
        self.models = []
        np.random.seed(seed)
        seeds = np.multiply(np.random.rand(self.n_estimators), 1000)
        seeds = seeds.tolist()
        seeds = map(int, seeds)

        label_col = "outcome"
        df = df.copy()
        df[label_col] = labels.values
        for i, bag_seed in enumerate(seeds):
            # sample
            df_train_undersampled = sample_by(
                df, label_col, self.d_ratios, seed=seed
            )
            # prepare dataset
            Y = df_train_undersampled[label_col]
            X = df_train_undersampled.drop([label_col], axis=1)
            # fit
            print("Model %i trained" % i)
            model = copy.deepcopy(self.classifier.fit(X, Y))
            self.models.append(model)

    def predict_proba(self, X):
        """
         Method that predicts the probability score for each row in X. It
        takes the mean of the score of each classifier.

        Args:
            X (array of float): dataset containing the features

        Returns: np.array of floats containing the predictions for each
         row in X
        """
        self.predictions = []
        for model in self.models:
            self.predictions.append(model.predict_proba(X))
        return np.mean(self.predictions, axis=0)


def sample_by(df, label_col, d_ratios, seed=1234):
    """
    Method sampling a dataframe by the value in one column
    Args:
        df (pd.DataFrame): dataframe to be sampled
        label_col (str): column name containing the key to sample on
        d_ratios (dict of int): dictionary containing the sampling frac
            for each class (i.e {0: 0.5, 1: 1})
        seed(int): sampling seed

    Returns (pd.DataFrame): sampled dataframe

    """
    sampled_df = pd.DataFrame()
    for key, value in d_ratios.items():
        temp_df = (
            df[df[label_col] == key]
            .sample(frac=value, random_state=seed)
            .reset_index(drop=True)
        )
        sampled_df = pd.concat([sampled_df, temp_df], ignore_index=True)
    return sampled_df.sample(frac=1).reset_index(drop=True)
