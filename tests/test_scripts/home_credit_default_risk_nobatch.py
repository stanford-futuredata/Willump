# Data files are too large to include.  Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
# Code source:  https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

import argparse
import pickle
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from home_credit_default_risk_utils import *
from willump.evaluation.willump_executor import willump_execute

warnings.simplefilter(action='ignore', category=FutureWarning)

base_folder = "tests/test_resources/home_credit_default_risk/"


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()


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


@willump_execute(disable=args.disable)
def join_and_lgbm(df, bureau, prev, pos, ins, cc, clf):
    df = df.to_frame().T
    df = df.merge(bureau, how='left', on='SK_ID_CURR')
    df = df.merge(prev, how='left', on='SK_ID_CURR')
    df = df.merge(pos, how='left', on='SK_ID_CURR')
    df = df.merge(ins, how='left', on='SK_ID_CURR')
    df = df.merge(cc, how='left', on='SK_ID_CURR')
    feats = [f for f in df.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    valid_x = df[feats]
    valid_x = valid_x.values
    oof_preds_proba = willump_predict_proba_function(clf, valid_x)
    return oof_preds_proba


def main(debug=False):
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    bureau, prev, pos, ins, cc = pickle.load(open(base_folder + "tables.csv", "rb"))
    clf = pickle.load(open(base_folder + "lgbm_model.pk", "rb"))
    df = df[df['TARGET'].notnull()]
    _, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    entry_list = []
    for i in valid_df.index:
        entry_list.append(valid_df.loc[i])
    del df
    mini_df = valid_df.iloc[0]
    join_and_lgbm(mini_df, bureau, prev, pos, ins, cc, clf)
    join_and_lgbm(mini_df, bureau, prev, pos, ins, cc, clf)
    oof_preds = []
    with timer("Joins and Prediction"):
        for entry in tqdm(entry_list):
            preds = join_and_lgbm(entry, bureau, prev, pos, ins, cc, clf)
            oof_preds.append(preds)

    oof_preds = np.hstack(oof_preds)
    print('Full AUC score %.6f' % willump_score_function(valid_df['TARGET'], oof_preds))


if __name__ == "__main__":
    with timer("Full model run"):
        main()
