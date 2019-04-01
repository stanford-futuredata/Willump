import argparse
import pickle
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from redis import StrictRedis as redis
from tqdm import tqdm

from willump.evaluation.willump_executor import willump_execute
from wsdm_utilities import *

model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
parser.add_argument("-a", "--caching", type=float, help="Max cache size")
parser.add_argument("-s", "--costly_statements", help="Mark costly (remotely stored) statements?", action="store_true")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    cascades = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_training_cascades.pk", "rb"))
    cascade_threshold = args.cascades

db = redis()


def get_row_to_merge_features_uf(key):
    global num_queries
    num_queries += 1
    redis_key = str(0) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_features_sf(key):
    global num_queries
    num_queries += 1
    redis_key = str(1) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_cluster_one(key):
    global num_queries
    return cluster_one.loc[key]


def get_row_to_merge_cluster_two(key):
    global num_queries
    return cluster_two.loc[key]


def get_row_to_merge_cluster_three(key):
    global num_queries
    return cluster_three.loc[key]


def get_row_to_merge_us_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(5) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_uc_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(2) + "_" + str(key)
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ac_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(4) + "_" + str(key)
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_sc_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(3) + "_" + str(key)
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_ss_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(6) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_as_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(7) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_gs_features(key):
    global num_queries
    return gs_features.loc[key]


def get_row_to_merge_cs_features(key):
    global num_queries
    return cs_features.loc[key]


def get_row_to_merge_ages_features(key):
    global num_queries
    return ages_features.loc[key]


def get_row_to_merge_ls_features(key):
    global num_queries
    return ls_features.loc[key]


def get_row_to_merge_gender_features(key):
    global num_queries
    return gender_features.loc[key]


def get_row_to_merge_composer_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(8) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_lyrs_features(key):
    global num_queries
    num_queries += 1
    redis_key = str(9) + "_" + str(int(key))
    ser_value = db.get(redis_key)
    value = pickle.loads(ser_value)
    return value


def get_row_to_merge_sns_features(key):
    global num_queries
    return sns_features.loc[key]


def get_row_to_merge_stabs_features(key):
    global num_queries
    return stabs_features.loc[key]


def get_row_to_merge_stypes_features(key):
    global num_queries
    return stypes_features.loc[key]


def get_row_to_merge_regs_features(key):
    global num_queries
    return regs_features.loc[key]


remote_funcs = ["get_row_to_merge_features_uf", "get_row_to_merge_features_sf", "get_row_to_merge_uc_features",
                "get_row_to_merge_sc_features", "get_row_to_merge_ac_features", "get_row_to_merge_us_features",
                "get_row_to_merge_ss_features", "get_row_to_merge_as_features",
                "get_row_to_merge_composer_features", "get_row_to_merge_lyrs_features"]

if args.caching is not None:
    cached_funcs = remote_funcs
    max_cache_size = args.caching
else:
    cached_funcs = ()
    max_cache_size = None

if args.costly_statements:
    costly_statements = remote_funcs
else:
    costly_statements = ()


@willump_execute(disable=args.disable, batch=False, cached_funcs=cached_funcs, costly_statements=costly_statements,
                 eval_cascades=cascades, cascade_threshold=cascade_threshold, max_cache_size=max_cache_size)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col, us_features, us_col, ss_features, ss_col, as_features,
             as_col, gs_features, gs_col, cs_features, cs_col, ages_features, ages_col, ls_features, ls_col,
             gender_features, gender_col, comps_features, comps_col, lyrs_features, lyrs_col, snames_features,
             snames_col, stabs_features, stabs_col, stypes_features, stypes_col, regs_features, regs_col):
    cluster_one_entry = combi[join_col_cluster_one]
    cluster_one_row = get_row_to_merge_cluster_one(cluster_one_entry)
    cluster_two_entry = combi[join_col_cluster_two]
    cluster_two_row = get_row_to_merge_cluster_two(cluster_two_entry)
    cluster_three_entry = combi[join_col_cluster_three]
    cluster_three_row = get_row_to_merge_cluster_three(cluster_three_entry)
    join_entry_one = combi[join_col_one]
    features_one_row = get_row_to_merge_features_uf(join_entry_one)
    join_entry_two = combi[join_col_two]
    features_two_row = get_row_to_merge_features_sf(join_entry_two)
    uc_entry = cluster_one_row[uc_join_col]
    uc_row = get_row_to_merge_uc_features(uc_entry)
    sc_entry = cluster_two_row[sc_join_col]
    sc_row = get_row_to_merge_sc_features(sc_entry)
    ac_entry = cluster_three_row[ac_join_col]
    ac_row = get_row_to_merge_ac_features(ac_entry)
    us_entry = combi[us_col]
    us_row = get_row_to_merge_us_features(us_entry)
    ss_entry = combi[ss_col]
    ss_row = get_row_to_merge_ss_features(ss_entry)
    as_entry = combi[as_col]
    as_row = get_row_to_merge_as_features(as_entry)
    gs_entry = combi[gs_col]
    gs_row = get_row_to_merge_gs_features(gs_entry)
    cs_entry = combi[cs_col]
    cs_row = get_row_to_merge_cs_features(cs_entry)
    ages_entry = combi[ages_col]
    ages_row = get_row_to_merge_ages_features(ages_entry)
    ls_entry = combi[ls_col]
    ls_row = get_row_to_merge_ls_features(ls_entry)
    gender_entry = combi[gender_col]
    gender_row = get_row_to_merge_gender_features(gender_entry)
    comps_entry = combi[comps_col]
    comps_row = get_row_to_merge_composer_features(comps_entry)
    lyrs_entry = combi[lyrs_col]
    lyrs_row = get_row_to_merge_lyrs_features(lyrs_entry)
    snames_entry = combi[snames_col]
    snames_row = get_row_to_merge_sns_features(snames_entry)
    stabs_entry = combi[stabs_col]
    stabs_row = get_row_to_merge_stabs_features(stabs_entry)
    stypes_entry = combi[stypes_col]
    stypes_row = get_row_to_merge_stypes_features(stypes_entry)
    regs_entry = combi[regs_col]
    regs_row = get_row_to_merge_regs_features(regs_entry)
    combi = pd.concat([features_one_row, features_two_row, uc_row, sc_row, ac_row, us_row,
                       ss_row, as_row, gs_row, cs_row, ages_row, ls_row, gender_row, comps_row,
                       lyrs_row, snames_row, stabs_row, stypes_row, regs_row], axis=0)
    combi = combi[FEATURES]
    combi = combi.values
    combi = combi.reshape(1, -1)
    preds = model.predict(combi)
    return preds


def add_features_and_predict(folder, combi):
    global features_uf, features_sf, cluster_one, cluster_two, cluster_three, uc_features, sc_features, \
        ac_features, us_features, ss_features, as_features, gs_features, cs_features, ages_features, ls_features, \
        gender_features, composer_features, lyrs_features, sns_features, stabs_features, stypes_features, regs_features
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_uf = features_uf.drop([join_col_uf], axis=1)
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)
    cluster_one, join_col_cluster_one = add_cluster(folder, col='msno', size=UC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_two, join_col_cluster_two = add_cluster(folder, col='song_id', size=SC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_three, join_col_cluster_three = add_cluster(folder, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features_eval(folder, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    uc_features = uc_features.set_index("cluster_msno_25")
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features_eval(folder, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    sc_features = sc_features.set_index("cluster_song_id_25")
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features_eval(folder, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    ac_features = ac_features.set_index("cluster_artist_name_25")
    # USER FEATURES
    us_features, us_col = scol_features_eval(folder, 'msno', 'u_')
    us_features = us_features.set_index("msno")
    # SONG FEATURES
    ss_features, ss_col = scol_features_eval(folder, 'song_id', 's_')
    ss_features = ss_features.set_index("song_id")
    # ARTIST FEATURES
    as_features, as_col = scol_features_eval(folder, 'artist_name', 'a_')
    as_features = as_features.set_index("artist_name")
    # GENRE FEATURES
    gs_features, gs_col = scol_features_eval(folder, 'genre_max', 'gmax_')
    gs_features = gs_features.set_index("genre_max")
    # CITY FEATURES
    cs_features, cs_col = scol_features_eval(folder, 'city', 'c_')
    cs_features = cs_features.set_index("city")
    # AGE FEATURES
    ages_features, ages_col = scol_features_eval(folder, 'bd', 'age_')
    ages_features = ages_features.set_index("bd")
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features_eval(folder, 'language', 'lang_')
    ls_features = ls_features.set_index("language")
    # GENDER FEATURES
    gender_features, gender_col = scol_features_eval(folder, 'gender', 'gen_')
    gender_features = gender_features.set_index("gender")
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features_eval(folder, 'composer', 'comp_')
    composer_features = composer_features.set_index("composer")
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features_eval(folder, 'lyricist', 'ly_')
    lyrs_features = lyrs_features.set_index("lyricist")
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features_eval(folder, 'source_screen_name', 'sn_')
    sns_features = sns_features.set_index("source_screen_name")
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features_eval(folder, 'source_system_tab', 'sst_')
    stabs_features = stabs_features.set_index("source_system_tab")
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features_eval(folder, 'source_type', 'st_')
    stypes_features = stypes_features.set_index("source_type")
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features_eval(folder, 'registered_via', 'rv_')
    regs_features = regs_features.set_index("registered_via")

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
    compile_one = do_merge(combi.iloc[0].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_features, cs_col,
                           ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                           composer_col,
                           lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                           stypes_col, regs_features, regs_col)
    compile_two = do_merge(combi.iloc[0].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_features, cs_col,
                           ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                           composer_col,
                           lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                           stypes_col, regs_features, regs_col)

    entry_list = []
    for i in range(num_rows):
        entry_list.append(combi.iloc[i])
    print("Copies created")
    preds = []
    start = time.time()
    for entry in tqdm(entry_list):
        pred = do_merge(entry, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                        join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                        join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                        ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                        gs_col, cs_features, cs_col,
                        ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                        composer_col,
                        lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                        stypes_col, regs_features, regs_col)
        preds.append(pred)
    elapsed_time = time.time() - start

    print('Latent feature join in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    return preds


def create_featureset(folder):
    global num_queries
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])
    y = combi["target"].values
    num_queries = 0

    combi_train, combi_valid, y_train, y_valid = train_test_split(combi, y, test_size=0.33, random_state=42)
    # Add features and predict.
    y_pred = np.hstack(add_features_and_predict(folder, combi_train))
    print("Train AUC: %f" % roc_auc_score(y_train, y_pred))
    print("Valid: Number of \"remote\" queries made: %d  Requests per row:  %f" %
          (num_queries, num_queries / len(y_pred)))
    num_queries = 0

    y_pred = np.hstack(add_features_and_predict(folder, combi_valid))
    print("Valid AUC: %f" % roc_auc_score(y_valid, y_pred))
    print("Valid: Number of \"remote\" queries made: %d  Requests per row:  %f" %
          (num_queries, num_queries / len(y_pred)))


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
