# Original source: https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977
# Data files can be found on Kaggle:  https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection

import argparse
import pickle
import time

from redis import StrictRedis as redis
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from adtracking_fraud_detection_util import *
from willump.evaluation.willump_executor import willump_execute

debug = True

base_folder = "tests/test_resources/adtracking_fraud_detection/"

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-a", "--caching", type=float, help="Max cache size")
parser.add_argument("-r", "--redis", help="Redis IP", type=str)
args = parser.parse_args()
if args.cascades:
    cascades = pickle.load(open(base_folder + "cascades.pk", "rb"))
else:
    cascades = None

if args.redis is None:
    redis_ip = "127.0.0.1"
else:
    redis_ip = args.redis

db = redis(host=redis_ip)


def get_row_to_merge_X_ip_channel(key):
    global num_queries
    num_queries += 1
    redis_key = str(0)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_ip_day_hour(key):
    global num_queries
    num_queries += 1
    redis_key = str(1)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_ip_app(key):
    global num_queries
    num_queries += 1
    redis_key = str(2)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_ip_app_os(key):
    global num_queries
    num_queries += 1
    redis_key = str(3)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_ip_device(key):
    global num_queries
    num_queries += 1
    redis_key = str(4)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_app_channel(key):
    global num_queries
    num_queries += 1
    redis_key = str(5)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_X_ip_device_app_os(key):
    global num_queries
    num_queries += 1
    redis_key = str(6)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_app_os(key):
    global num_queries
    num_queries += 1
    redis_key = str(7)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_day_hour(key):
    global num_queries
    num_queries += 1
    redis_key = str(8)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_app(key):
    global num_queries
    num_queries += 1
    redis_key = str(9)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_day_hour_channel(key):
    global num_queries
    num_queries += 1
    redis_key = str(10)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_app_channel_var_day(key):
    global num_queries
    num_queries += 1
    redis_key = str(11)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_app_os_hour(key):
    global num_queries
    num_queries += 1
    redis_key = str(12)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ip_app_chl_mean_hour(key):
    global num_queries
    num_queries += 1
    redis_key = str(13)
    for attribute in key:
        redis_key += "_" + str(int(attribute))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


remote_funcs = ["get_row_to_merge_X_ip_channel", "get_row_to_merge_X_ip_day_hour", "get_row_to_merge_X_ip_app"
    , "get_row_to_merge_X_ip_app_os", "get_row_to_merge_X_ip_device", "get_row_to_merge_X_app_channel",
                "get_row_to_merge_X_ip_device_app_os", "get_row_to_merge_ip_app_os", "get_row_to_merge_ip_day_hour",
                "get_row_to_merge_ip_app", "get_row_to_merge_ip_day_hour_channel",
                "get_row_to_merge_ip_app_channel_var_day"
    , "get_row_to_merge_ip_app_os_hour", "get_row_to_merge_ip_app_chl_mean_hour"]

if args.caching is not None:
    cached_funcs = remote_funcs
    max_cache_size = args.caching
else:
    cached_funcs = ()
    max_cache_size = None


@willump_execute(disable=args.disable, batch=False, eval_cascades=cascades,
                 cached_funcs=cached_funcs, max_cache_size=max_cache_size, async_funcs=remote_funcs)
def process_input_and_predict(input_df):
    X_ip_channel_entry = tuple(input_df[X_ip_channel_jc])
    X_ip_channel_row = get_row_to_merge_X_ip_channel(X_ip_channel_entry)
    X_ip_day_hour_entry = tuple(input_df[X_ip_day_hour_jc])
    X_ip_day_hour_row = get_row_to_merge_X_ip_day_hour(X_ip_day_hour_entry)
    X_ip_app_entry = tuple(input_df[X_ip_app_jc])
    X_ip_app_row = get_row_to_merge_X_ip_app(X_ip_app_entry)
    X_ip_app_os_entry = tuple(input_df[X_ip_app_os_jc])
    X_ip_app_os_row = get_row_to_merge_X_ip_app_os(X_ip_app_os_entry)
    X_ip_device_entry = tuple(input_df[X_ip_device_jc])
    X_ip_device_row = get_row_to_merge_X_ip_device(X_ip_device_entry)
    X_app_channel_entry = tuple(input_df[X_app_channel_jc])
    X_app_channel_row = get_row_to_merge_X_app_channel(X_app_channel_entry)
    X_ip_device_app_os_entry = tuple(input_df[X_ip_device_app_os_jc])
    X_ip_device_app_os_row = get_row_to_merge_X_ip_device_app_os(X_ip_device_app_os_entry)
    ip_app_os_entry = tuple(input_df[ip_app_os_jc])
    ip_app_os_row = get_row_to_merge_ip_app_os(ip_app_os_entry)
    ip_day_hour_entry = tuple(input_df[ip_day_hour_jc])
    ip_day_hour_row = get_row_to_merge_ip_day_hour(ip_day_hour_entry)
    ip_app_entry = tuple(input_df[ip_app_jc])
    ip_app_row = get_row_to_merge_ip_app(ip_app_entry)
    ip_day_hour_channel_entry = tuple(input_df[ip_day_hour_channel_jc])
    ip_day_hour_channel_row = get_row_to_merge_ip_day_hour_channel(ip_day_hour_channel_entry)
    ip_app_channel_var_day_entry = tuple(input_df[ip_app_channel_var_day_jc])
    ip_app_channel_var_day_row = get_row_to_merge_ip_app_channel_var_day(ip_app_channel_var_day_entry)
    ip_app_os_hour_entry = tuple(input_df[ip_app_os_hour_jc])
    ip_app_os_hour_row = get_row_to_merge_ip_app_os_hour(ip_app_os_hour_entry)
    ip_app_chl_mean_hour_entry = tuple(input_df[ip_app_chl_mean_hour_jc])
    ip_app_chl_mean_hour_row = get_row_to_merge_ip_app_chl_mean_hour(ip_app_chl_mean_hour_entry)
    combined_df = pd.concat([input_df, X_ip_channel_row, X_ip_day_hour_row, X_ip_app_row, X_ip_app_os_row,
                             X_ip_device_row, X_app_channel_row, X_ip_device_app_os_row, ip_app_os_row,
                             ip_day_hour_row, ip_app_row, ip_day_hour_channel_row,
                             ip_app_channel_var_day_row, ip_app_os_hour_row, ip_app_chl_mean_hour_row], axis=0)
    combined_df = combined_df[predictors]
    combined_df = combined_df.values
    combined_df = combined_df.reshape(1, -1)
    preds = willump_predict_function(clf, combined_df)
    return preds


if __name__ == "__main__":
    clf = pickle.load(open(base_folder + "model.pk", "rb"))

    max_rows = 10000000
    debug_size = 100000
    if debug:
        train_start_point = 0
        train_end_point = debug_size
    else:
        train_start_point = 0
        train_end_point = max_rows

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32',
    }

    train_df = pd.read_csv(base_folder + "train.csv", parse_dates=['click_time'], skiprows=range(1, train_start_point),
                           nrows=train_end_point - train_start_point,
                           dtype=dtypes,
                           usecols=['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed'])

    len_train = len(train_df)
    gc.collect()
    tables_filename = base_folder + "aggregate_tables%s%s.pk" % (train_start_point, train_end_point)
    (X_ip_channel, X_ip_channel_jc, X_ip_day_hour, X_ip_day_hour_jc, X_ip_app, X_ip_app_jc,
     X_ip_app_os, X_ip_app_os_jc,
     X_ip_device, X_ip_device_jc, X_app_channel, X_app_channel_jc, X_ip_device_app_os, X_ip_device_app_os_jc,
     ip_app_os, ip_app_os_jc, ip_day_hour, ip_day_hour_jc, ip_app, ip_app_jc, ip_day_hour_channel,
     ip_day_hour_channel_jc, ip_app_channel_var_day, ip_app_channel_var_day_jc, ip_app_os_hour,
     ip_app_os_hour_jc, ip_app_chl_mean_hour, ip_app_chl_mean_hour_jc, nextClick, nextClick_shift, X1, X7) = \
        pickle.load(open(tables_filename, "rb"))

    X_ip_channel = X_ip_channel.set_index(X_ip_channel_jc)
    X_ip_day_hour = X_ip_day_hour.set_index(X_ip_day_hour_jc)
    X_ip_app = X_ip_app.set_index(X_ip_app_jc)
    X_ip_app_os = X_ip_app_os.set_index(X_ip_app_os_jc)
    X_ip_device = X_ip_device.set_index(X_ip_device_jc)
    X_app_channel = X_app_channel.set_index(X_app_channel_jc)
    X_ip_device_app_os = X_ip_device_app_os.set_index(X_ip_device_app_os_jc)
    ip_app_os = ip_app_os.set_index(ip_app_os_jc)
    ip_day_hour = ip_day_hour.set_index(ip_day_hour_jc)
    ip_app = ip_app.set_index(ip_app_jc)
    ip_day_hour_channel = ip_day_hour_channel.set_index(ip_day_hour_channel_jc)
    ip_app_channel_var_day = ip_app_channel_var_day.set_index(ip_app_channel_var_day_jc)
    ip_app_os_hour = ip_app_os_hour.set_index(ip_app_os_hour_jc)
    ip_app_chl_mean_hour = ip_app_chl_mean_hour.set_index(ip_app_chl_mean_hour_jc)

    train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
    train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
    train_df["nextClick"] = nextClick
    train_df["nextClick_shift"] = nextClick_shift
    train_df["X1"] = X1
    train_df["X7"] = X7
    train_df = train_df.drop(columns=["click_time"])

    remote_df_list = [X_ip_channel, X_ip_day_hour, X_ip_app, X_ip_app_os, X_ip_device, X_app_channel,
                      X_ip_device_app_os, ip_app_os, ip_day_hour, ip_app, ip_day_hour_channel,
                      ip_app_channel_var_day, ip_app_os_hour, ip_app_chl_mean_hour]
    for i, entry in enumerate(remote_df_list):
        remote_df = entry
        for key in remote_df.index:
            value = remote_df.loc[key]
            ser_index_value = pickle.dumps(value)
            if isinstance(key, tuple):
                redis_key = str(i)
                for attribute in key:
                    redis_key += "_" + str(attribute)
            else:
                redis_key = str(i) + "_" + str(key)
            db.set(redis_key, ser_index_value)

    target = 'is_attributed'
    predictors = ['nextClick', 'nextClick_shift', 'app', 'device', 'os', 'channel', 'hour', 'day', 'ip_tcount',
                  'ip_tchan_count', 'ip_app_count', 'ip_app_os_count', 'ip_app_os_var', 'ip_app_channel_var_day',
                  'ip_app_channel_mean_hour', 'X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8']
    categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

    train_y = train_df[target].values

    train_df, valid_df, _, valid_y = train_test_split(train_df, train_y, test_size=0.1, shuffle=False)
    del train_y

    if debug:
        train_num_rows = 20000
        valid_num_rows = 10000
    else:
        train_num_rows = len(train_df)
        valid_num_rows = len(valid_df)

    num_queries = 0
    mini_df = valid_df.iloc[0]
    process_input_and_predict(mini_df)
    process_input_and_predict(mini_df)

    entry_list = []
    for i in range(train_num_rows):
        entry_list.append(train_df.iloc[i])
    y_preds = []
    num_queries = 0
    start = time.time()
    for entry in tqdm(entry_list):
        pred = process_input_and_predict(entry)
        y_preds.append(pred)
    elapsed_time = time.time() - start
    y_preds = np.hstack(y_preds)
    print('Train prediction in %f seconds rows %d throughput %f Number Requests %d Requests per Row %f: ' % (
        elapsed_time, train_num_rows, train_num_rows / elapsed_time, num_queries, num_queries / train_num_rows))

    trained_cache = globals()["__willump_cache"]
    pickle.dump(trained_cache, open(base_folder + "trained_cache.pk", "wb"))
