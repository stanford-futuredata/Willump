import argparse
import pickle
import time

from sklearn import metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from willump.evaluation.willump_executor import willump_execute
from wsdm_utilities import *


def auc_score(y_valid, y_pred):
    fpr, tpr, thresholds = metrics.roc_curve(y_valid, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


model = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_model.pk", "rb"))


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--cascades", type=float, help="Cascade threshold")
parser.add_argument("-d", "--disable", help="Disable Willump", action="store_true")
args = parser.parse_args()
if args.cascades is None:
    cascades = None
    cascade_threshold = 1.0
else:
    assert (0.5 <= args.cascades <= 1.0)
    cascades = pickle.load(open("tests/test_resources/wsdm_cup_features/wsdm_training_cascades.pk", "rb"))
    cascade_threshold = args.cascades


@willump_execute(batch=False, disable=args.disable, eval_cascades=cascades, cascade_threshold=cascade_threshold)
def do_merge(combi, features_one, join_col_one, features_two, join_col_two, cluster_one, join_col_cluster_one,
             cluster_two, join_col_cluster_two, cluster_three, join_col_cluster_three, uc_features, uc_join_col,
             sc_features, sc_join_col, ac_features, ac_join_col, us_features, us_col, ss_features, ss_col, as_features,
             as_col, gs_features, gs_col, cs_features, cs_col, ages_features, ages_col, ls_features, ls_col,
             gender_features, gender_col, comps_features, comps_col, lyrs_features, lyrs_col, snames_features,
             snames_col, stabs_features, stabs_col, stypes_features, stypes_col, regs_features, regs_col):
    combi = combi.to_frame().T
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
    combi = combi[FEATURES]
    combi = combi.values
    preds = willump_predict_function(model, combi)
    return preds


def add_features_and_predict(folder, combi):
    features_uf, join_col_uf = load_als_dataframe(folder, size=UF_SIZE, user=True, artist=False)
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
    preds = []
    print("Copies created")
    times = []
    for entry in tqdm(entry_list):
        t0 = time.time()
        pred = do_merge(entry, features_uf, join_col_uf, features_sf, join_col_sf, cluster_one,
                        join_col_cluster_one, cluster_two, join_col_cluster_two, cluster_three,
                        join_col_cluster_three, uc_features, uc_join_col, sc_features, sc_join_col, ac_features,
                        ac_join_col, us_features, us_col, ss_features, ss_col, as_features, as_col, gs_features,
                        gs_col, cs_feature, cs_col,
                        ages_features, ages_col, ls_features, ls_col, gender_features, gender_col, composer_features,
                        composer_col,
                        lyrs_features, lyrs_col, sns_features, sns_col, stabs_features, stabs_col, stypes_features,
                        stypes_col, regs_features, regs_col)
        time_elapsed = time.time() - t0
        times.append(time_elapsed)
        preds.append(pred)
    p50 = np.percentile(times, 50)
    p99 = np.percentile(times, 99)

    print("p50 Latency: %f p99 Latency: %f" %
          (p50, p99))

    return preds


def create_featureset(folder):
    combi = load_combi_prep(folder=folder, split=None)
    combi = combi.dropna(subset=["target"])
    y = combi["target"].values

    _, combi, _, y = train_test_split(combi, y, test_size=0.33, random_state=42)

    # add latent features
    y_pred = np.hstack(add_features_and_predict(folder, combi))
    print("Train AUC: %f" % willump_score_function(y, y_pred))


if __name__ == '__main__':
    create_featureset("tests/test_resources/wsdm_cup_features/")
