import pytest
from sklearn.dummy import DummyClassifier
from myml.classifiers.balancedbagging import RatioBaggingClassifier, BaggingClassifier


@pytest.mark.parametrize(
    argnames='classifier, d_ratios, prediction',
    argvalues=[
        (DummyClassifier(strategy='most_frequent'), {0: 1, 1: 1},  0),
        (DummyClassifier(strategy='constant', constant=1), {0: 1, 1: 1}, 1),
        (DummyClassifier(strategy='most_frequent'), {0: 0.1, 1: 1},  1),
    ]
)
def test_bagging_classifier(
        classifier, prediction, d_ratios, df_random_labelled_two_dim_with_ratio):
    df = df_random_labelled_two_dim_with_ratio(
        num_rows=150, num_features=40, ratio_positives=0.5, outcome_col='outcome')
    Y = df.outcome
    X = df.drop('outcome', axis=1)
    model = BaggingClassifier(classifier, d_ratios=d_ratios, n_estimators=5)
    model.fit(X, Y)
    prob = model.predict_proba(X)
    assert prob[:, 1].mean() == prediction


@pytest.mark.parametrize(
    argnames='classifier, ratios, prediction',
    argvalues=[
        (DummyClassifier(strategy='most_frequent'), ([1, 0.8, 0.6], [1, 1, 1]), 0),
        (DummyClassifier(strategy='most_frequent'), ([0.1, 0.1, 0.1], [1, 1, 1]), 1),
        (DummyClassifier(strategy='constant', constant=1), ([1, 0.8, 0.6], [1, 1, 1]),
         1)
    ]
)
def test_ratiobagging_classifier(
        classifier, ratios, prediction, df_random_labelled_two_dim_with_ratio):
    df = df_random_labelled_two_dim_with_ratio(
        num_rows=150, num_features=40, ratio_positives=0.2, outcome_col='outcome')
    Y = df.outcome
    X = df.drop('outcome', axis=1)
    model = RatioBaggingClassifier(classifier, ratios)
    model.fit(X, Y)
    prob = model.predict_proba(X)
    assert prob[:, 1].mean() == prediction

