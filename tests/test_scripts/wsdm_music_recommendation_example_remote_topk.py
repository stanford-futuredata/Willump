import argparse
import pickle
import time

import redis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from willump.evaluation.willump_executor import willump_execute
from wsdm_utilities import *

model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", action="store_true", help="Cascades?")
parser.add_argument("-k", "--top_k", type=int, help="Top-K to return", required=True)
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
parser.add_argument("-r", "--redis", help="Redis IP", type=str)
args = parser.parse_args()

if args.cascades:
    cascades = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_training_cascades.pk", "rb"))
else:
    cascades = None

top_K = args.top_k

if args.redis is None:
    redis_ip = "127.0.0.1"
else:
    redis_ip = args.redis

db = redis.StrictRedis(host=redis_ip)


def get_row_to_merge_features_uf(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(0) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_features_sf(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(1) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_cluster_one(keys):
    global num_queries
    return cluster_one.loc[keys].reset_index()


def get_row_to_merge_cluster_two(keys):
    global num_queries
    return cluster_two.loc[keys].reset_index()


def get_row_to_merge_cluster_three(keys):
    global num_queries
    return cluster_three.loc[keys].reset_index()


def get_row_to_merge_us_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(5) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_uc_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(2) + "_" + str(key)
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_ac_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(4) + "_" + str(key)
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_sc_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(3) + "_" + str(key)
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_ss_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(6) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_as_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(7) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_gs_features(keys):
    global num_queries
    return gs_features.loc[keys].reset_index()


def get_row_to_merge_cs_features(keys):
    global num_queries
    return cs_features.loc[keys].reset_index()


def get_row_to_merge_ages_features(keys):
    global num_queries
    return ages_features.loc[keys].reset_index()


def get_row_to_merge_ls_features(keys):
    global num_queries
    return ls_features.loc[keys].reset_index()


def get_row_to_merge_gender_features(keys):
    global num_queries
    return gender_features.loc[keys].reset_index()


def get_row_to_merge_composer_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(8) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_lyrs_features(keys):
    global num_queries
    num_queries += len(keys)
    pipe = db.pipeline()
    for key in keys:
        redis_key = str(9) + "_" + str(int(key))
        pipe.get(redis_key)
    serialized_result = pipe.execute()
    pre_result = []
    for ser_entry in serialized_result:
        pre_result.append(pickle.loads(ser_entry))
    result = pd.concat(pre_result, axis=1).T
    result = result.reset_index().drop("index", axis=1)
    return result


def get_row_to_merge_sns_features(keys):
    global num_queries
    return sns_features.loc[keys].reset_index()


def get_row_to_merge_stabs_features(keys):
    global num_queries
    return stabs_features.loc[keys].reset_index()


def get_row_to_merge_stypes_features(keys):
    global num_queries
    return stypes_features.loc[keys].reset_index()


def get_row_to_merge_regs_features(keys):
    global num_queries
    return regs_features.loc[keys].reset_index()


remote_funcs = ["get_row_to_merge_features_uf", "get_row_to_merge_features_sf", "get_row_to_merge_uc_features",
                "get_row_to_merge_sc_features", "get_row_to_merge_ac_features", "get_row_to_merge_us_features",
                "get_row_to_merge_ss_features", "get_row_to_merge_as_features",
                "get_row_to_merge_composer_features", "get_row_to_merge_lyrs_features"]

all_funcs = ["get_row_to_merge_features_uf", "get_row_to_merge_features_sf", "get_row_to_merge_uc_features",
             "get_row_to_merge_sc_features", "get_row_to_merge_ac_features", "get_row_to_merge_us_features",
             "get_row_to_merge_ss_features", "get_row_to_merge_as_features",
             "get_row_to_merge_composer_features", "get_row_to_merge_lyrs_features",
             "get_row_to_merge_gs_features", "get_row_to_merge_cs_features", "get_row_to_merge_ages_features",
             "get_row_to_merge_ls_features", "get_row_to_merge_gender_features", "get_row_to_merge_sns_features",
             "get_row_to_merge_stabs_features", "get_row_to_merge_stypes_features", "get_row_to_merge_regs_features"]


@willump_execute(disable=args.disable, eval_cascades=cascades, top_k=top_K, async_funcs=remote_funcs)
def do_merge(combi):
    cluster_one_entry = combi[join_col_cluster_one].values
    cluster_one_row = get_row_to_merge_cluster_one(cluster_one_entry)
    cluster_two_entry = combi[join_col_cluster_two].values
    cluster_two_row = get_row_to_merge_cluster_two(cluster_two_entry)
    cluster_three_entry = combi[join_col_cluster_three].values
    cluster_three_row = get_row_to_merge_cluster_three(cluster_three_entry)
    join_entry_one = combi[join_col_uf].values
    features_one_row = get_row_to_merge_features_uf(join_entry_one)
    join_entry_two = combi[join_col_sf].values
    features_two_row = get_row_to_merge_features_sf(join_entry_two)
    uc_entry = cluster_one_row[uc_join_col].values
    uc_row = get_row_to_merge_uc_features(uc_entry)
    sc_entry = cluster_two_row[sc_join_col].values
    sc_row = get_row_to_merge_sc_features(sc_entry)
    ac_entry = cluster_three_row[ac_join_col].values
    ac_row = get_row_to_merge_ac_features(ac_entry)
    us_entry = combi[us_col].values
    us_row = get_row_to_merge_us_features(us_entry)
    ss_entry = combi[ss_col].values
    ss_row = get_row_to_merge_ss_features(ss_entry)
    as_entry = combi[as_col].values
    as_row = get_row_to_merge_as_features(as_entry)
    gs_entry = combi[gs_col].values
    gs_row = get_row_to_merge_gs_features(gs_entry)
    cs_entry = combi[cs_col].values
    cs_row = get_row_to_merge_cs_features(cs_entry)
    ages_entry = combi[ages_col].values
    ages_row = get_row_to_merge_ages_features(ages_entry)
    ls_entry = combi[ls_col].values
    ls_row = get_row_to_merge_ls_features(ls_entry)
    gender_entry = combi[gender_col].values
    gender_row = get_row_to_merge_gender_features(gender_entry)
    comps_entry = combi[composer_col].values
    comps_row = get_row_to_merge_composer_features(comps_entry)
    lyrs_entry = combi[lyrs_col].values
    lyrs_row = get_row_to_merge_lyrs_features(lyrs_entry)
    snames_entry = combi[sns_col].values
    snames_row = get_row_to_merge_sns_features(snames_entry)
    stabs_entry = combi[stabs_col].values
    stabs_row = get_row_to_merge_stabs_features(stabs_entry)
    stypes_entry = combi[stypes_col].values
    stypes_row = get_row_to_merge_stypes_features(stypes_entry)
    regs_entry = combi[regs_col].values
    regs_row = get_row_to_merge_regs_features(regs_entry)
    combi = pd.concat([features_one_row, features_two_row, uc_row, sc_row, ac_row, us_row,
                       ss_row, as_row, gs_row, cs_row, ages_row, ls_row, gender_row, comps_row,
                       lyrs_row, snames_row, stabs_row, stypes_row, regs_row], axis=1)
    combi = combi[FEATURES]
    preds = willump_predict_proba_function(model, combi)
    return preds


def add_features_and_predict(folder, combi):
    global features_uf, features_sf, cluster_one, cluster_two, cluster_three, uc_features, sc_features, \
        ac_features, us_features, ss_features, as_features, gs_features, cs_features, ages_features, ls_features, \
        gender_features, composer_features, lyrs_features, sns_features, stabs_features, stypes_features, regs_features, \
        join_col_uf, join_col_sf, join_col_cluster_one, join_col_cluster_two, join_col_cluster_three, uc_join_col, \
        sc_join_col, ac_join_col, us_col, ss_col, as_col, gs_col, cs_col, ages_col, ls_col, gender_col, \
        composer_col, lyrs_col, sns_col, stabs_col, stypes_col, regs_col
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_uf = features_uf.set_index(join_col_uf).astype("float64")
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)
    features_sf = features_sf.set_index(join_col_sf).astype("float64")
    cluster_one, join_col_cluster_one = add_cluster(folder, col='msno', size=UC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_two, join_col_cluster_two = add_cluster(folder, col='song_id', size=SC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_three, join_col_cluster_three = add_cluster(folder, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features_eval(folder, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    uc_features = uc_features.set_index("cluster_msno_25").astype("float64")
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features_eval(folder, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    sc_features = sc_features.set_index("cluster_song_id_25").astype("float64")
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features_eval(folder, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    ac_features = ac_features.set_index("cluster_artist_name_25").astype("float64")
    # USER FEATURES
    us_features, us_col = scol_features_eval(folder, 'msno', 'u_')
    us_features = us_features.set_index("msno").astype("float64")
    # SONG FEATURES
    ss_features, ss_col = scol_features_eval(folder, 'song_id', 's_')
    ss_features = ss_features.set_index("song_id").astype("float64")
    # ARTIST FEATURES
    as_features, as_col = scol_features_eval(folder, 'artist_name', 'a_')
    as_features = as_features.set_index("artist_name").astype("float64")
    # GENRE FEATURES
    gs_features, gs_col = scol_features_eval(folder, 'genre_max', 'gmax_')
    gs_features = gs_features.set_index("genre_max").astype("float64")
    # CITY FEATURES
    cs_features, cs_col = scol_features_eval(folder, 'city', 'c_')
    cs_features = cs_features.set_index("city").astype("float64")
    # AGE FEATURES
    ages_features, ages_col = scol_features_eval(folder, 'bd', 'age_')
    ages_features = ages_features.set_index("bd").astype("float64")
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features_eval(folder, 'language', 'lang_')
    ls_features = ls_features.set_index("language").astype("float64")
    # GENDER FEATURES
    gender_features, gender_col = scol_features_eval(folder, 'gender', 'gen_')
    gender_features = gender_features.set_index("gender").astype("float64")
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features_eval(folder, 'composer', 'comp_')
    composer_features = composer_features.set_index("composer").astype("float64")
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features_eval(folder, 'lyricist', 'ly_')
    lyrs_features = lyrs_features.set_index("lyricist").astype("float64")
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features_eval(folder, 'source_screen_name', 'sn_')
    sns_features = sns_features.set_index("source_screen_name").astype("float64")
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features_eval(folder, 'source_system_tab', 'sst_')
    stabs_features = stabs_features.set_index("source_system_tab").astype("float64")
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features_eval(folder, 'source_type', 'st_')
    stypes_features = stypes_features.set_index("source_type").astype("float64")
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features_eval(folder, 'registered_via', 'rv_')
    regs_features = regs_features.set_index("registered_via").astype("float64")

    remote_df_list = [
        features_uf,
        features_sf,
        uc_features,
        sc_features,
        ac_features,
        us_features,
        ss_features,
        as_features,
        composer_features,
        lyrs_features,
    ]

    for i, entry in enumerate(remote_df_list):
        df = entry
        for key in df.index:
            value = df.loc[key]
            ser_value = pickle.dumps(value)
            redis_key = str(i) + "_" + str(key)
            db.set(redis_key, ser_value)

    num_rows = len(combi)
    orig_preds = do_merge(combi)
    do_merge(combi.iloc[0:3].copy())

    start = time.time()
    preds = do_merge(combi)
    elapsed_time = time.time() - start

    orig_model_top_k_idx = np.argsort(orig_preds)[-1 * top_K:]
    actual_model_top_k_idx = np.argsort(preds)[-1 * top_K:]
    precision = len(np.intersect1d(orig_model_top_k_idx, actual_model_top_k_idx)) / top_K

    orig_model_sum = sum(orig_preds[orig_model_top_k_idx])
    actual_model_sum = sum(preds[actual_model_top_k_idx])

    print("Title Processing Time %fs Num Rows %d Throughput %f rows/sec" %
          (elapsed_time, num_rows, num_rows / elapsed_time))
    print("Precision: %f Orig Model Sum: %f Actual Model Sum: %f" % (precision, orig_model_sum, actual_model_sum))

    return preds


def create_featureset(folder):
    global num_queries
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"]).astype("float64")
    num_queries = 0

    _, combi_valid = train_test_split(combi, test_size=0.2, random_state=42)
    # Add features and predict.
    add_features_and_predict(folder, combi_valid)


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
