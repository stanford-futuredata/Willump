# Data files are too large to include.  Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
# Code source:  https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

import gc
import pickle
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from willump.evaluation.willump_executor import willump_execute

warnings.simplefilter(action='ignore', category=FutureWarning)

base_folder = "tests/test_resources/home_credit_default_risk/"


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.5f}s".format(title, time.time() - t0))


# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category=True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns=categorical_columns, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


# Preprocess application_train.csv and application_test.csv
def application_train_test(num_rows=None, nan_as_category=False):
    # Read data and merge
    df = pd.read_csv(base_folder + 'application_train.csv', nrows=num_rows)
    print("Train samples: {}".format(len(df)))
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    df = df[df['CODE_GENDER'] != 'XNA']

    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df, nan_as_category)

    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    # Some simple new features (percentages)
    df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
    df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    df['PAYMENT_RATE'] = df['AMT_ANNUITY'] / df['AMT_CREDIT']
    return df


@willump_execute()
def join_and_lgbm(df, bureau, prev, pos, ins, cc, feats, clf):
    df = df.merge(bureau, how='left', on='SK_ID_CURR')
    df = df.merge(prev, how='left', on='SK_ID_CURR')
    df = df.merge(pos, how='left', on='SK_ID_CURR')
    df = df.merge(ins, how='left', on='SK_ID_CURR')
    df = df.merge(cc, how='left', on='SK_ID_CURR')
    valid_x = df[feats]
    oof_preds_proba = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)
    oof_preds = oof_preds_proba[:, 1]
    return oof_preds


def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    bureau, prev, pos, ins, cc, feats = pickle.load(open(base_folder + "tables.csv", "rb"))
    bureau = bureau.reset_index()
    prev = prev.reset_index()
    pos = pos.reset_index()
    ins = ins.reset_index()
    cc = cc.reset_index()
    clf = pickle.load(open(base_folder + "lgbm_model.pk", "rb"))
    df = df[df['TARGET'].notnull()]
    _, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    del df
    join_and_lgbm(valid_df, bureau, prev, pos, ins, cc, feats, clf)
    join_and_lgbm(valid_df, bureau, prev, pos, ins, cc, feats, clf)
    with timer("Joins and Prediction"):
        oof_preds = join_and_lgbm(valid_df, bureau, prev, pos, ins, cc, feats, clf)

    print('Full AUC score %.6f' % roc_auc_score(valid_df['TARGET'], oof_preds))


if __name__ == "__main__":
    with timer("Full model run"):
        main()
