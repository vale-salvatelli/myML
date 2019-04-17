"""
This module contains multi-stage classifiers
"""
from __future__ import division
import numpy as np
import pandas as pd


class StackingClassifier(object):
    """
    Class the defines a fit and predict_proba method for a stacking classifier.
    It takes in input a list of trained models and build a classifier on the
    predictions of the trained models.
    """

    def __init__(self, trained_models, classifier):
        self.classifier = classifier
        self.base_models = trained_models

    def fit(self, X, Y):
        scores = []
        for model in self.base_models:
            predictions = model.predict_proba(X)[:, 1]
            scores.append(predictions)
        new_X = pd.DataFrame(np.transpose(scores))
        self.model = self.classifier.fit(new_X, Y)

    def predict_proba(self, X):
        return self.model.predict_proba(X)