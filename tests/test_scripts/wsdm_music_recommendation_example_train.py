# Original Source: https://github.com/rn5l/wsdm-cup-2018-music

import argparse
import pickle
import time

from sklearn.model_selection import train_test_split

import willump.evaluation.willump_executor
from wsdm_utilities import *

training_cascades = {}

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--costly_statements", help="Mark costly (remotely stored) statements?", action="store_true")
args = parser.parse_args()

if args.costly_statements:
    costly_statements = ("features_one", "features_two", "uc_features",
                         "sc_features", "ac_features", "us_features",
                         "ss_features", "as_features",
                         "comps_features", "lyrs_features")
else:
    costly_statements = ()


@willump.evaluation.willump_executor.willump_execute(batch=True, training_cascades=training_cascades,
                                                     costly_statements=costly_statements,
                                                     willump_train_function=willump_train_function,
                                                     willump_predict_function=willump_predict_function,
                                                     willump_score_function=willump_score_function,
                                                     willump_predict_proba_function=willump_predict_proba_function)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col, us_features, us_col, ss_features, ss_col, as_features,
             as_col, gs_features, gs_col, cs_features, cs_col, ages_features, ages_col, ls_features, ls_col,
             gender_features, gender_col, comps_features, comps_col, lyrs_features, lyrs_col, snames_features,
             snames_col, stabs_features, stabs_col, stypes_features, stypes_col, regs_features, regs_col, y_train):
    combi = combi.merge(cluster_one, how='left', on=join_col_cluster_one)
    combi = combi.merge(cluster_two, how='left', on=join_col_cluster_two)
    combi = combi.merge(cluster_three, how='left', on=join_col_cluster_three)
    combi = combi.merge(features_one, how='left', on=join_col_one)
    combi = combi.merge(features_two, how='left', on=join_col_two)
    combi = combi.merge(uc_features, how='left', on=uc_join_col)
    combi = combi.merge(sc_features, how='left', on=sc_join_col)
    combi = combi.merge(ac_features, how='left', on=ac_join_col)
    combi = combi.merge(us_features, how='left', on=us_col)
    combi = combi.merge(ss_features, how='left', on=ss_col)
    combi = combi.merge(as_features, how='left', on=as_col)
    combi = combi.merge(gs_features, how='left', on=gs_col)
    combi = combi.merge(cs_features, how='left', on=cs_col)
    combi = combi.merge(ages_features, how='left', on=ages_col)
    combi = combi.merge(ls_features, how='left', on=ls_col)
    combi = combi.merge(gender_features, how='left', on=gender_col)
    combi = combi.merge(comps_features, how='left', on=comps_col)
    combi = combi.merge(lyrs_features, how='left', on=lyrs_col)
    combi = combi.merge(snames_features, how='left', on=snames_col)
    combi = combi.merge(stabs_features, how='left', on=stabs_col)
    combi = combi.merge(stypes_features, how='left', on=stypes_col)
    combi = combi.merge(regs_features, how='left', on=regs_col)
    train_data = combi[FEATURES]
    train_data = train_data.fillna(0)
    model = willump_train_function(train_data, y_train)
    return model


def scol_features(folder, combi, col, prefix):
    tmp = pd.DataFrame()
    group = combi.groupby([col])
    group_pos = combi[combi.target == 1].groupby(col)
    tmp[prefix + 'played'] = group.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = group_pos.size().astype(np.int32)
    tmp[prefix + 'played_pos'] = tmp[prefix + 'played_pos'].fillna(0)
    tmp[prefix + 'played_rel'] = (tmp[prefix + 'played'] / tmp[prefix + 'played'].max()).astype(np.float32)
    tmp[prefix + 'played_rel_global'] = (tmp[prefix + 'played'] / len(combi)).astype(np.float32)
    tmp[prefix + 'played_pos_rel'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played_pos'].max()).astype(np.float32)
    tmp[prefix + 'played_ratio'] = (tmp[prefix + 'played_pos'] / tmp[prefix + 'played']).astype(np.float32)
    tmp[col] = tmp.index
    csv_name = folder + prefix + 'cluster' + '_' + col + '_features' + '.csv'
    tmp.to_csv(csv_name)
    return tmp, col


def add_features_and_train_model(folder, combi):
    y = combi["target"].values

    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
    features_sf, join_col_sf = load_als_dataframe(folder, size=SF_SIZE, user=False, artist=False)
    cluster_one, join_col_cluster_one = add_cluster(folder, col='msno', size=UC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_two, join_col_cluster_two = add_cluster(folder, col='song_id', size=SC_SIZE, overlap=True, positive=True,
                                                    content=False)
    cluster_three, join_col_cluster_three = add_cluster(folder, col='artist_name', size=SC_SIZE, overlap=True,
                                                        positive=True, content=False)
    uc_combi = combi.merge(cluster_one, how='left', on=join_col_cluster_one)
    uc_combi = uc_combi.merge(cluster_two, how='left', on=join_col_cluster_two)
    uc_combi = uc_combi.merge(cluster_three, how='left', on=join_col_cluster_three)
    # USER CLUSTER FEATURES
    uc_features, uc_join_col = scol_features(folder, uc_combi, 'cluster_msno_' + str(UC_SIZE), 'uc_')
    uc_features = uc_features.reset_index(drop=True)
    # SONG CLUSTER FEATURES
    sc_features, sc_join_col = scol_features(folder, uc_combi, 'cluster_song_id_' + str(SC_SIZE), 'sc_')
    sc_features = sc_features.reset_index(drop=True)
    # ARTIST CLUSTER FEATURES
    ac_features, ac_join_col = scol_features(folder, uc_combi, 'cluster_artist_name_' + str(UC_SIZE), 'ac_')
    ac_features = ac_features.reset_index(drop=True)
    # USER FEATURES
    us_features, us_col = scol_features(folder, combi, 'msno', 'u_')
    us_features = us_features.reset_index(drop=True)
    # SONG FEATURES
    ss_features, ss_col = scol_features(folder, combi, 'song_id', 's_')
    ss_features = ss_features.reset_index(drop=True)
    # ARTIST FEATURES
    as_features, as_col = scol_features(folder, combi, 'artist_name', 'a_')
    as_features = as_features.reset_index(drop=True)
    # GENRE FEATURES
    gs_features, gs_col = scol_features(folder, combi, 'genre_max', 'gmax_')
    gs_features = gs_features.reset_index(drop=True)
    # CITY FEATURES
    cs_feature, cs_col = scol_features(folder, combi, 'city', 'c_')
    cs_feature = cs_feature.reset_index(drop=True)
    # AGE FEATURES
    ages_features, ages_col = scol_features(folder, combi, 'bd', 'age_')
    ages_features = ages_features.reset_index(drop=True)
    # LANGUAGE FEATURES
    ls_features, ls_col = scol_features(folder, combi, 'language', 'lang_')
    ls_features = ls_features.reset_index(drop=True)
    # GENDER FEATURES
    gender_features, gender_col = scol_features(folder, combi, 'gender', 'gen_')
    gender_features = gender_features.reset_index(drop=True)
    # COMPOSER FEATURES
    composer_features, composer_col = scol_features(folder, combi, 'composer', 'comp_')
    composer_features = composer_features.reset_index(drop=True)
    # LYRICIST FEATURES
    lyrs_features, lyrs_col = scol_features(folder, combi, 'lyricist', 'ly_')
    lyrs_features = lyrs_features.reset_index(drop=True)
    # SOURCE NAME FEATURES
    sns_features, sns_col = scol_features(folder, combi, 'source_screen_name', 'sn_')
    sns_features = sns_features.reset_index(drop=True)
    # SOURCE TAB FEATURES
    stabs_features, stabs_col = scol_features(folder, combi, 'source_system_tab', 'sst_')
    stabs_features = stabs_features.reset_index(drop=True)
    # SOURCE TYPE FEATURES
    stypes_features, stypes_col = scol_features(folder, combi, 'source_type', 'st_')
    stypes_features = stypes_features.reset_index(drop=True)
    # SOURCE TYPE FEATURES
    regs_features, regs_col = scol_features(folder, combi, 'registered_via', 'rv_')
    regs_features = regs_features.reset_index(drop=True)

    combi, _, y, _ = train_test_split(combi, y, test_size=0.2, random_state=42)

    num_rows = len(combi)

    start = time.time()
    model = do_merge(combi, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                     join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                     join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                     ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                     gs_col, cs_feature, cs_col,
                     ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                     composer_col,
                     lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                     stypes_col, regs_features, regs_col, y)
    elapsed_time = time.time() - start

    print('First (Python) training in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    start = time.time()
    do_merge(combi, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
             join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
             join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
             ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
             gs_col, cs_feature, cs_col,
             ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
             composer_col,
             lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
             stypes_col, regs_features, regs_col, y)
    elapsed_time = time.time() - start

    print('Second (Willump) training in %f seconds rows %d throughput %f: ' % (
        elapsed_time, num_rows, num_rows / elapsed_time))

    return model


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])

    model = add_features_and_train_model(folder, combi)

    pickle.dump(model, open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "wb"))
    pickle.dump(training_cascades, open("tests/test_resources/wsdm_cup_features/wsdm_training_cascades.pk", "wb"))


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
