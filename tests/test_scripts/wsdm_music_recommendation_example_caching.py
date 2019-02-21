import pickle
import time

from sklearn import metrics

import willump.evaluation.willump_executor
from wsdm_utilities import *
from tqdm import tqdm


def auc_score(y_valid, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))
model.coef_ = np.ones(64).reshape(1, -1)

FEATURES = LATENT_SONG_FEATURES + LATENT_USER_FEATURES
num_queries = 0


def get_row_to_merge_features_uf(key):
    global num_queries
    num_queries += 1
    return features_uf.loc[key]


def get_row_to_merge_features_sf(key):
    global num_queries
    num_queries += 1
    return features_sf.loc[key]


@willump.evaluation.willump_executor.willump_execute(cached_funcs=("get_row_to_merge_features_uf",
                                                                   "get_row_to_merge_features_sf"))
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col, us_features, us_col, ss_features, ss_col, as_features,
             as_col, gs_features, gs_col, cs_features, cs_col, ages_features, ages_col, ls_features, ls_col,
             gender_features, gender_col, comps_features, comps_col, lyrs_features, lyrs_col, snames_features,
             snames_col, stabs_features, stabs_col, stypes_features, stypes_col, regs_features, regs_col):
    join_entry_one = combi[join_col_one]
    features_one_row = get_row_to_merge_features_uf(join_entry_one)
    join_entry_two = combi[join_col_two]
    features_two_row = get_row_to_merge_features_sf(join_entry_two)
    combi = pd.concat([features_one_row, features_two_row], axis=0)
    combi = combi[FEATURES]
    combi = combi.values
    combi = combi.reshape(1, -1)
    preds = model.predict(combi)
    return preds


def add_features_and_predict(folder, combi):
    global features_uf, features_sf
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
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features_eval(folder, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features_eval(folder, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    # USER FEATURES
    us_features, us_col = scol_features_eval(folder, 'msno', 'u_')
    # SONG FEATURES
    ss_features, ss_col = scol_features_eval(folder, 'song_id', 's_')
    # ARTIST FEATURES
    as_features, as_col = scol_features_eval(folder, 'artist_name', 'a_')
    # GENRE FEATURES
    gs_features, gs_col = scol_features_eval(folder, 'genre_max', 'gmax_')
    # CITY FEATURES
    cs_feature, cs_col = scol_features_eval(folder, 'city', 'c_')
    # AGE FEATURES
    ages_features, ages_col = scol_features_eval(folder, 'bd', 'age_')
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features_eval(folder, 'language', 'lang_')
    # GENDER FEATURES
    gender_features, gender_col = scol_features_eval(folder, 'gender', 'gen_')
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features_eval(folder, 'composer', 'comp_')
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features_eval(folder, 'lyricist', 'ly_')
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features_eval(folder, 'source_screen_name', 'sn_')
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features_eval(folder, 'source_system_tab', 'sst_')
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features_eval(folder, 'source_type', 'st_')
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features_eval(folder, 'registered_via', 'rv_')

    num_rows = len(combi)
    compile_one = do_merge(combi.iloc[0].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_feature, cs_col,
                           ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                           composer_col,
                           lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                           stypes_col, regs_features, regs_col)
    compile_two = do_merge(combi.iloc[0].copy(), features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                           join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                           join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                           ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                           gs_col, cs_feature, cs_col,
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
                        gs_col, cs_feature, cs_col,
                        ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                        composer_col,
                        lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                        stypes_col, regs_features, regs_col)
        preds.append(pred)
    elapsed_time = time.time() - start

    print('Latent feature join in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    print("Number of \"remote\" queries made: %d" % num_queries)

    return preds


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])
    y = combi["target"].values
    # Add features and predict.
    y_pred = add_features_and_predict(folder, combi)
    print("Train AUC: %f" % auc_score(y, y_pred))


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
