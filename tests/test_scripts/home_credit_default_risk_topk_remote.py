# Data files are too large to include.  Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
# Code source:  https://www.kaggle.com/jsaguiar/lightgbm-with-simple-features

import argparse
import pickle
import time
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import redis
from sklearn.model_selection import train_test_split

from home_credit_default_risk_utils import *
from willump.evaluation.willump_executor import willump_execute

warnings.simplefilter(action='ignore', category=FutureWarning)

base_folder = "tests/test_resources/home_credit_default_risk/"

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--top_k_cascade", type=int, help="Top-K to return")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-r", "--redis", help="Redis IP", type=str)
args = parser.parse_args()
if args.top_k_cascade is None:
    cascades = None
    top_K = None
else:
    cascades = pickle.load(open(base_folder + "training_cascades.pk", "rb"))
    top_K = args.top_k_cascade

if args.redis is None:
    redis_ip = "127.0.0.1"
else:
    redis_ip = args.redis

db = redis.StrictRedis(host=redis_ip)


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


def get_row_to_merge_features_bureau(keys):
    global num_queries, num_nan
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(0) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        try:
            pre_result.append(pickle.loads(ser_entry))
        except TypeError:
            pre_result.append(bureau_nan)
            num_nan += 1
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_features_prev(keys):
    global num_queries, num_nan
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(1) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        try:
            pre_result.append(pickle.loads(ser_entry))
        except TypeError:
            pre_result.append(prev_nan)
            num_nan += 1
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_features_pos(keys):
    global num_queries, num_nan
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(2) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        try:
            pre_result.append(pickle.loads(ser_entry))
        except TypeError:
            pre_result.append(pos_nan)
            num_nan += 1
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_features_ins(keys):
    global num_queries, num_nan
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(3) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        try:
            pre_result.append(pickle.loads(ser_entry))
        except TypeError:
            pre_result.append(ins_nan)
            num_nan += 1
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_features_cc(keys):
    global num_queries, num_nan
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(4) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        try:
            pre_result.append(pickle.loads(ser_entry))
        except TypeError:
            pre_result.append(cc_nan)
            num_nan += 1
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


remote_funcs = ["get_row_to_merge_features_bureau",
                "get_row_to_merge_features_prev",
                "get_row_to_merge_features_pos",
                "get_row_to_merge_features_ins",
                "get_row_to_merge_features_cc"]


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K, async_funcs=remote_funcs)
def join_and_lgbm(df):
    join_entry_bureau = df['SK_ID_CURR'].values
    bureau_df = get_row_to_merge_features_bureau(join_entry_bureau)
    join_entry_prev = df['SK_ID_CURR'].values
    prev_df = get_row_to_merge_features_prev(join_entry_prev)
    join_entry_pos = df['SK_ID_CURR'].values
    pos_df = get_row_to_merge_features_pos(join_entry_pos)
    join_entry_ins = df['SK_ID_CURR'].values
    ins_df = get_row_to_merge_features_ins(join_entry_ins)
    join_entry_cc = df['SK_ID_CURR'].values
    cc_df = get_row_to_merge_features_cc(join_entry_cc)
    df = pd.concat([df, bureau_df, prev_df, pos_df, ins_df, cc_df], axis=1)
    feats = [f for f in df.columns if
             f not in ['TARGET', 'SK_ID_CURR', 'SK_ID_BUREAU', 'SK_ID_PREV', 'index']]
    valid_x = df[feats]
    oof_preds_proba = willump_predict_proba_function(clf, valid_x)
    return oof_preds_proba


def main(debug=False):
    global bureau, prev, pos, ins, cc, clf, num_queries, bureau_nan, prev_nan, pos_nan, ins_nan, cc_nan, num_nan
    num_rows = 10000 if debug else None
    df = application_train_test(num_rows)
    bureau, prev, pos, ins, cc = pickle.load(open(base_folder + "tables.csv", "rb"))

    bureau_nan = bureau.iloc[0].copy().drop("SK_ID_CURR")
    for entry in bureau_nan.index:
        bureau_nan[entry] = np.NaN
    prev_nan = prev.iloc[0].copy().drop("SK_ID_CURR")
    for entry in prev_nan.index:
        prev_nan[entry] = np.NaN
    pos_nan = pos.iloc[0].copy().drop("SK_ID_CURR")
    for entry in pos_nan.index:
        pos_nan[entry] = np.NaN
    ins_nan = ins.iloc[0].copy().drop("SK_ID_CURR")
    for entry in ins_nan.index:
        ins_nan[entry] = np.NaN
    cc_nan = cc.iloc[0].copy().drop("SK_ID_CURR")
    for entry in cc_nan.index:
        cc_nan[entry] = np.NaN

    remote_df_list = [
        bureau,
        prev,
        pos,
        ins,
        cc,
    ]

    for i, remote_df in enumerate(remote_df_list):
        remote_df = remote_df.set_index("SK_ID_CURR")
        for key in remote_df.index:
            value = remote_df.loc[key]
            ser_value = pickle.dumps(value)
            redis_key = str(i) + "_" + str(int(key))
            db.set(redis_key, ser_value)

    num_queries = 0
    num_nan = 0

    clf = pickle.load(open(base_folder + "lgbm_model.pk", "rb"))
    df = df[df['TARGET'].notnull()]
    _, valid_df = train_test_split(df, test_size=0.2, random_state=42)
    del df

    valid_df = valid_df.reset_index().drop("index", axis=1)

    mini_df = valid_df.iloc[0:3]
    join_and_lgbm(mini_df)
    join_and_lgbm(mini_df)
    with timer("Joins and Prediction"):
        oof_preds = join_and_lgbm(valid_df)
    print("Num Queries: %d  Num NaN: %d Queries per Row: %f" % (num_queries, num_nan, num_queries / len(valid_df)))

    top_k_idx = np.argsort(oof_preds)[-1 * top_K:]
    top_k_values = [oof_preds[i] for i in top_k_idx]
    sum_values = 0

    for idx, value in zip(top_k_idx, top_k_values):
        print(idx, value)
        sum_values += value

    print("Sum of top %d values: %f" % (top_K, sum_values))

    out_filename = "credit_top%d.pk" % top_K
    pickle.dump(oof_preds, open(out_filename, "wb"))


if __name__ == "__main__":
    with timer("Full model run"):
        main()
