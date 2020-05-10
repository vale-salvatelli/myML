"""
This module contains ensemble binary classifiers designed for
imbalanced datasets
"""
import numpy as np
import pandas as pd
import copy


class RatioBaggingClassifier(object):
    """
    Class that defines a fit and predict_proba method for an ensemble classifier
    that combines classifiers of the same type trained at different positives to
    negatives ratio. The final score is given by the mean of the scores.
    """

    def __init__(self, classifier, ratios=([1, 0.8, 0.6], [1, 1, 1])):
        """

        Args:
            classifier: base classifier, sklearn-like
            ratios (tuple of int list): desired sampling ratio for the two classes.
                The first list containing the ratios for the 0 class, the second
                list containing the ratios for the 1 class. The length of the 2 list
                will correspond to the number of classifiers combined
        """
        self.classifier = classifier
        self.ratios = ratios
        self.models = []
        self.predictions = []

    def fit(self, df, seed=1234, label_col='outcome'):
        """
        Method that trains several models, each on a sample of df that has imbalance
        ratio as described in ratios.

        Args:
            df (pd.DataFrame): df containing features and labels
            seed (int): seed for the sampling
            label_col (str): name of the column containing the label

        """
        self.models = []

        if len(self.ratios[0]) != len(self.ratios[1]):
            raise ValueError(
                'ratios has to be of the form ([1,0.8,0.6],[1,1,1])'
            )

        for i in range(len(self.ratios[0])):
            # sample
            d_ratios = {0: self.ratios[0][i], 1:self.ratios[1][i]}
            df = df.sample(frac=1).reset_index(drop=True)
            df_train_undersampled = sample_by(
                df, label_col, d_ratios, seed=seed)

            # prepare dataset
            Y = df_train_undersampled[label_col]
            X = df_train_undersampled.drop([label_col], axis=1)
            # fit
            print("Model %i" % (i + 1))
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
    of the scores. For each base classifier, all the positives are
    included, the negatives are bootstrapped from the full dataset according
    to the P2Nratio. If sample_feature <1, a different sample of features
    is passed to each bag.
    """

    def __init__(self, classifier, n_estimators=10, d_ratios={0:0.1, 1:1},
                 sample_feature=1):
        """

        Args:
            classifier: base classifier, sklearn-like
            n_estimators (int): number of classifier to be combined
            d_ratios (dict): sampling ratio for each class
            sample_feature (float): percentage of features to be randomly
                selected in each classifier
        """
        self.classifier = classifier
        self.n_estimators = n_estimators
        self.d_ratios = d_ratios
        self.sample_feature = sample_feature
        self.models = []
        self.predictions = []

    def fit(self, df, seed=1234, label_col='outcome'):
        """

        Args:
          df (pd.DataFrame): df containing features and labels
          seed (int): seed to initialize the sampling
          label_col (str): name of the column containing the label
        """
        self.models = []
        np.random.seed(self.seed)
        seeds = np.multiply(np.random.rand(self.n_estimators), 1000)
        seeds = seeds.tolist()
        seeds = map(int, seeds)

        for bag_seed in seeds:
            # sample
            df_train_undersampled = sample_by(
                df, label_col, self.d_ratios, seed=seed)
            # prepare dataset
            Y = df_train_undersampled[label_col]
            X = df_train_undersampled.drop([label_col], axis=1)
            all_cols = list(X)
            if self.sample_feature < 1:
                n_feat = int(X.shape[1] * self.sample_feature)
                X = X.sample(n=n_feat, axis=1)
                dropped_cols = list(set(all_cols) - set(list(X)))
                for col in dropped_cols:
                    X[col] = np.nan
                if len(X.columns) != len(all_cols):
                    raise ValueError(
                        'subsampled dataframe must have all the cols'
                    )

            # fit
            print("Model %i" % bag_seed)
            model = copy.deepcopy(self.classifier.fit(X[all_cols], Y))
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
    for key, value in d_ratios:
        temp_df = df[df[label_col] == key].\
            sample(frac=value, random_state=seed).reset_index(drop=True)
        sampled_df = pd.concat([sampled_df, temp_df])
    return sampled_df
