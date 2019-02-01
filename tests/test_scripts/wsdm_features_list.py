# LATENT VECTOR SIZES
UF_SIZE = 32
UF2_SIZE = 32
SF_SIZE = 32
AF_SIZE = 32
# CLUSTER SIZES
UC_SIZE = 25
SC_SIZE = 25
USC_SIZE = 25

LATENT_USER_FEATURES = ['uf_' + str(i) for i in range(UF_SIZE)]
LATENT_SONG_FEATURES = ['sf_' + str(i) for i in range(SF_SIZE)]
UC_FEATURES = [
    'uc_played',
    'uc_played_pos',
    'uc_played_rel',
    'uc_played_rel_global',
    'uc_played_pos_rel',
    'uc_played_ratio'
]
SC_FEATURES = [
    'sc_played',
    'sc_played_pos',
    'sc_played_rel',
    'sc_played_rel_global',
    'sc_played_ratio'
]

AC_FEATURES = [
    'ac_played',
    'ac_played_pos',
    'ac_played_rel',
    'ac_played_rel_global',
    'ac_played_ratio'
]
AGE_FEATURES = [
    'age_played',
    'age_played_pos',
    'age_played_rel',
    'age_played_rel_global',
    'age_played_pos_rel',
    'age_played_ratio'
]

REG_VIA_FEATURES = [
    'rv_played',
    'rv_played_pos',
    'rv_played_rel',
    'rv_played_rel_global',
    'rv_played_pos_rel',
    'rv_played_ratio'
]

LANGUAGE_FEATURES = [
    'lang_played',
    'lang_played_pos',
    'lang_played_rel',
    'lang_played_rel_global',
    'lang_played_pos_rel',
    'lang_played_ratio'
]

GENDER_FEATURES = [
    'gen_played',
    'gen_played_pos',
    'gen_played_rel',
    'gen_played_rel_global',
    'gen_played_pos_rel',
    'gen_played_ratio'
]
LYRICIST_FEATURES = [
    'ly_played',
    'ly_played_rel',
    'ly_played_rel_global',
]
COMPOSER_FEATURES = [
    'comp_played',
    'comp_played_rel',
    'comp_played_rel_global',
]

USER_FEATURES = [
    'u_played',
    'u_played_rel',
    'u_played_rel_global',
]

SONG_FEATURES = [
    's_played',
    's_played_rel',
    's_played_rel_global',
]

ARTIST_FEATURES = [
    'a_played',
    'a_played_rel',
    'a_played_rel_global',
]

SOURCE_NAME_FEATURES = [
    'sn_played',
    'sn_played_pos',
    'sn_played_rel',
    'sn_played_rel_global',
    'sn_played_pos_rel',
    'sn_played_ratio'
]

SOURCE_TAB_FEATURES = [
    'sst_played',
    'sst_played_pos',
    'sst_played_rel',
    'sst_played_rel_global',
    'sst_played_pos_rel',
    'sst_played_ratio'
]

SOURCE_TYPE_FEATURES = [
    'st_played',
    'st_played_pos',
    'st_played_rel',
    'st_played_pos_rel',
    'st_played_ratio'
]

CITY_FEATURES = [
    'c_played',
    'c_played_pos',
    'c_played_rel',
    'c_played_rel_global',
    'c_played_pos_rel',
    'c_played_ratio'
]

GENRE_FEATURES_MAX = [
    'gmax_played',
    'gmax_played_pos',
    'gmax_played_rel',
    'gmax_played_rel_global',
    'gmax_played_pos_rel',
    'gmax_played_ratio',
]

FEATURES = LATENT_SONG_FEATURES + LATENT_USER_FEATURES
FEATURES = FEATURES + SC_FEATURES
FEATURES = FEATURES + UC_FEATURES
FEATURES = FEATURES + AC_FEATURES

FEATURES = FEATURES + LANGUAGE_FEATURES  # pos included
FEATURES = FEATURES + AGE_FEATURES  # pos included
FEATURES = FEATURES + GENDER_FEATURES  # pos included
FEATURES = FEATURES + REG_VIA_FEATURES  #pos included
FEATURES = FEATURES + COMPOSER_FEATURES  # + COMPOSER_FEATURES_RATIO + COMPOSER_FEATURES_POS
FEATURES = FEATURES + LYRICIST_FEATURES  # + LYRICIST_FEATURES_RATIO #+ LYRICIST_FEATURES_POS

FEATURES = FEATURES + USER_FEATURES  # + USER_FEATURES_RATIO + USER_FEATURES_POS

FEATURES = FEATURES + SONG_FEATURES  # + SONG_FEATURES_RATIO #+ SONG_FEATURES_POS

FEATURES = FEATURES + ARTIST_FEATURES

FEATURES = FEATURES + GENRE_FEATURES_MAX

FEATURES = FEATURES + SOURCE_NAME_FEATURES  # pos included
FEATURES = FEATURES + SOURCE_TAB_FEATURES  # pos included
FEATURES = FEATURES + SOURCE_TYPE_FEATURES  # pos included
FEATURES = FEATURES + CITY_FEATURES  # pos included


