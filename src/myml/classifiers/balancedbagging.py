"""
This module contains ensemble classifiers designed for imbalanced datasets
"""
from __future__ import division
import numpy as np
import pandas as pd
import copy


class DifferentRatiosClassifier(object):
    """
    Class that defines a fit and predict_proba method for an ensemble classifier
    that combines classifiers of the same type trained at different positives to
    negatives ratio (P2Nratio). The final score is given by the mean of the scores.
    For each base classifier, all the positives are included, the negatives are
    bootstrapped from the full dataset according to the P2Nratio.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.models = []
        self.predictions = []
        self.seed = None
        self.d_ratios = None

    def fit(self, df, ratios=[[1,0.8,0.6],[1,1,1]], seed=1234, label_col='outcome'):
        """
        Method that trains several models, each on a sample of df that has imbalance
        ratio as described in ratios.

        Args:
            df (pd.DataFrame): df containing features and labels
            ratios (list of int list): desired sampling ratio for the two classes.
                The first list containing the ratios for the 0 class, the second
                list containing the ratios for the 1 class. The length of the 2 list
                will correspond to the number of classifiers combined
            seed (int): seed for the sampling
            label_col (str): name of the column containing the label

        """
        self.models = []
        self.seed = seed
        self.ratios = ratios

        if len(self.ratios[0]) != len(self.ratios[1]):
            raise ValueError(
                'ratios has to be of the form ([1,0.8,0.6],[1,1,1])'
            )

        for i in range(len(self.ratios[0])):
            # sample
            d_ratios = {0: self.ratios[0][i], 1:self.ratios[1][i]}
            df = df.sample(frac=1).reset_index(drop=True)
            df_train_undersampled = self.sample_by(df, label_col, d_ratios,
                                                   random_state=seed)

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

    @staticmethod
    def sample_by(df, label_col, d_ratios):
        """
        Method sampling a dataframe by the value in one column
        Args:
            df (pd.DataFrame): dataframe to be sampled
            label_col (str): column name containing the key to sample on
            d_ratios (dict of int): dictionary containing the sampling frac
                for each class (i.e {0: 0.5, 1: 1})

        Returns (pd.DataFrame): sampled dataframe

        """
        for key, value in d_ratios:
            temp_df = df[df[label_col] == key].sample(frac = value).\
                      reset_index(drop=True)
            sampled_df = pd.concat([sampled_df, temp_df])
        return sampled_df


class BaggingClassifier(object):
    """
    Class that defines a fit and predict_proba method for an ensemble classifier
    that combines classifiers of the same type trained on bootstrapped samples,
    at fixed positives to negatives ratio. The final score is given by the mean
    of the scores. For each base classifier, all the positives are included,
    the negatives are bootstrapped from the full dataset according to the P2Nratio.
    If sample_feature <1, a different sample of features is passed to
    each bag.
    """

    def __init__(self, classifier):
        self.classifier = classifier
        self.models = []

    def fit(self, sdf, n_estimators, ratios=(0.1, 0.1), seed=1234,
            label_col='outcome', sample_feature=1):
        """
        sample_feature is the sampling fraction i.e. sample_feature=0.7
        means that 70% of features will be randomly sampled.
        """
        self.seed = seed
        self.ratios = ratios
        self.neg_sample_ratios = ratios[0]
        self.pos_sample_ratios = ratios[1]

        np.random.seed(self.seed)
        seeds = np.multiply(np.random.rand(n_estimators), 1000)
        seeds = seeds.tolist()
        seeds = map(int, seeds)

        for bag_seed in seeds:
            # sample
            df_train_undersampled = sdf.sampleBy(
                label_col, fractions={0: self.neg_sample_ratios,
                                      1: self.pos_sample_ratios},
                seed=bag_seed).toPandas()
            # prepare dataset
            Y = df_train_undersampled[label_col]
            X = df_train_undersampled.drop([label_col], axis=1)
            self.all_cols = list(X)
            if sample_feature < 1:
                n_feat = int(X.shape[1] * sample_feature)
                X = X.sample(n=n_feat, axis=1)
                dropped_cols = list(set(self.all_cols) - set(list(X)))
                for col in dropped_cols:
                    if 'freq' in col:
                        X[col] = 0
                    else:
                        X[col] = np.nan
                if len(X.columns) != len(self.all_cols):
                    raise ValueError(
                        'subsampled dataframe must have all the cols'
                    )

            # fit
            print("Model %i" % bag_seed)
            model = copy.deepcopy(self.classifier.fit(X[self.all_cols], Y))
            self.models.append(model)

    def predict_proba(self, X):
        self.predictions_ = list()
        for model in self.models:
            self.predictions_.append(model.predict_proba(X))
        return np.mean(self.predictions_, axis=0)
