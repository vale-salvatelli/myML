#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for myml.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""

import pytest
import numpy as np
import pandas as pd


@pytest.fixture(scope='module')
def df_random_labelled_two_dim_with_ratio():
    def _df(num_rows, num_features, ratio_positives=0.5,
            outcome_col='outcome'):
        """ Provides a function which can be used to construct two dimensional
        dataframes with a given ratio of positives under the 'outcome'
        column (0.5 by default) and an optional RCS column
        Returns:
            pandas.DataFrame: function which returns a df with
            'patient_id', 'outcome' and other columns
        """
        num_pos = int(num_rows * ratio_positives)
        labels = np.concatenate((np.zeros((num_rows - num_pos, 1)),
                                 np.ones((num_pos, 1))))

        df = pd.DataFrame(np.random.randn(num_rows, num_features))
        df.columns = ['var_{}'.format(str(col)) for col in df.columns]
        df[outcome_col] = labels
        return df

    return _df