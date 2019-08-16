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

from home_credit_default_risk_utils import *
from willump.evaluation.willump_executor import willump_execute

warnings.simplefilter(action='ignore', category=FutureWarning)

base_folder = "tests/test_resources/home_credit_default_risk/"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-c", "--cascades", action="store_true", help="Cascades?")
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
parser.add_argument("-b", "--debug", help="Debug Mode", action="store_true")
args = parser.parse_args()

if args.cascades:
    cascades = pickle.load(open(base_folder + "training_cascades.pk", "rb"))
else:
    cascades = None
top_K = args.top_k


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


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K)
def join_and_lgbm(df, bureau, prev, pos, ins, cc, clf):
    df = df.merge(bureau, how='left', on='SK_ID_CURR')
    df = df.merge(prev, how='left', on='SK_ID_CURR')
    df = df.merge(pos, how='left', on='SK_ID_CURR')
    df = df.merge(ins, how='left', on='SK_ID_CURR')
    df = df.merge(cc, how='left', on='SK_ID_CURR')
    feats = [f for f in df.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    valid_x = df[feats]
    oof_preds_proba = willump_predict_proba_function(clf, valid_x)
    return oof_preds_proba


def main():
    num_rows = 1000 if args.debug else None
    df = application_train_test(num_rows)
    bureau, prev, pos, ins, cc = pickle.load(open(base_folder + "tables.csv", "rb"))
    clf = pickle.load(open(base_folder + "lgbm_model.pk", "rb"))
    df = df[df['TARGET'].notnull()]
    _, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    num_rows = len(valid_df)
    del df
    orig_preds = join_and_lgbm(valid_df, bureau, prev, pos, ins, cc, clf)
    mini_df = valid_df.iloc[0:3]
    join_and_lgbm(mini_df, bureau, prev, pos, ins, cc, clf)
    t0 = time.time()
    preds = join_and_lgbm(valid_df, bureau, prev, pos, ins, cc, clf)
    elapsed_time = time.time() - t0
    orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
    actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
    precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

    orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
    actual_model_sum = sum(preds[actual_model_top_k_idx])

    print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
          (elapsed_time, num_rows, num_rows / elapsed_time))
    print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))


if __name__ == "__main__":
    with timer("Full model run"):
        main()
