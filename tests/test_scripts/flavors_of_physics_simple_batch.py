# https://github.com/gramolin/flavours-of-physics

import numpy as np
import pandas as pd
import time
import willump.evaluation.willump_executor

# Physical constants:
# TODO:  Proper typing for c (and other floats mixed with arrays).
# c = 299.792458  # Speed of light
m_mu = 105.6583715  # Muon mass (in MeV)
m_tau = 1776.82  # Tau mass (in MeV)


# Function to add extra features:
@willump.evaluation.willump_executor.willump_execute
def add_features(df):
    # Number of events:
    N = len(df)

    # Read some of the original features:
    pt = df['pt'].values
    dira = df['dira'].values
    LifeTime = df['LifeTime'].values
    FlightDistance = df['FlightDistance'].values
    FlightDistanceError = df['FlightDistanceError'].values
    DOCAone = df['DOCAone'].values
    DOCAtwo = df['DOCAtwo'].values
    DOCAthree = df['DOCAthree'].values
    isolation_a = df['isolationa'].values
    isolation_b = df['isolationb'].values
    isolation_c = df['isolationc'].values
    isolation_d = df['isolationd'].values
    isolation_e = df['isolatione'].values
    isolation_f = df['isolationf'].values
    p0_p = df['p0_p'].values
    p1_p = df['p1_p'].values
    p2_p = df['p2_p'].values
    p0_pt = df['p0_pt'].values
    p1_pt = df['p1_pt'].values
    p2_pt = df['p2_pt'].values
    p0_eta = df['p0_eta'].values
    p1_eta = df['p1_eta'].values
    p2_eta = df['p2_eta'].values
    p0_IsoBDT = df['p0_IsoBDT'].values
    p1_IsoBDT = df['p1_IsoBDT'].values
    p2_IsoBDT = df['p2_IsoBDT'].values
    p0_track_Chi2Dof = df['p0_track_Chi2Dof'].values
    p1_track_Chi2Dof = df['p1_track_Chi2Dof'].values
    p2_track_Chi2Dof = df['p2_track_Chi2Dof'].values
    p0_IP = df['p0_IP'].values
    p1_IP = df['p1_IP'].values
    p2_IP = df['p2_IP'].values
    p0_IPSig = df['p0_IPSig'].values
    p1_IPSig = df['p1_IPSig'].values
    p2_IPSig = df['p2_IPSig'].values
    CDF1 = df['CDF1'].values
    CDF2 = df['CDF2'].values
    CDF3 = df['CDF3'].values

    # Differences between pseudorapidities of the final-state particles:
    eta_01 = p0_eta - p1_eta
    eta_02 = p0_eta - p2_eta
    eta_12 = p1_eta - p2_eta

    # Transverse collinearity of the final-state particles (equals to 1 if they are collinear):
    t_coll = (p0_pt + p1_pt + p2_pt) / pt

    # Longitudinal momenta of the final-state particles:
    p0_z = p0_pt * np.sinh(p0_eta)
    p1_z = p1_pt * np.sinh(p1_eta)
    p2_z = p2_pt * np.sinh(p2_eta)

    # Energies of the final-state particles:
    E0 = np.sqrt(np.square(m_mu) + np.square(p0_p))
    E1 = np.sqrt(np.square(m_mu) + np.square(p1_p))
    E2 = np.sqrt(np.square(m_mu) + np.square(p2_p))

    # Energy and momenta of the mother particle:
    E = E0 + E1 + E2
    pz = p0_z + p1_z + p2_z
    p = np.sqrt(np.square(pt) + np.square(pz))

    # Energies and momenta of the final-state particles relative to those of the mother particle:
    E0_ratio = E0 / E
    E1_ratio = E1 / E
    E2_ratio = E2 / E
    p0_pt_ratio = p0_pt / pt
    p1_pt_ratio = p1_pt / pt
    p2_pt_ratio = p2_pt / pt

    # Mass of the mother particle calculated from FlightDistance and LifeTime:
    beta_gamma = FlightDistance / (LifeTime * c)
    M_lt = p / beta_gamma

    # TODO:  Handle flag_M properly.
    flag_M = np.zeros(N)

    # Invariant mass of the mother particle calculated from its energy and momentum:
    M_inv = np.sqrt(np.square(E) - np.square(p))

    # Relativistic gamma and beta of the mother particle:
    gamma = E / M_inv
    beta = np.sqrt(np.square(gamma) - 1.) / gamma

    # Difference between M_lt and M_inv:
    Delta_M = M_lt - M_inv

    # Difference between energies of the mother particle calculated in two different ways:
    Delta_E = np.sqrt(np.square(M_lt) + np.square(p)) - E

    # Other extra features:
    FlightDistanceSig = FlightDistance / FlightDistanceError
    DOCA_sum = DOCAone + DOCAtwo + DOCAthree
    isolation_sum = isolation_a + isolation_b + isolation_c + isolation_d + isolation_e + isolation_f
    IsoBDT_sum = p0_IsoBDT + p1_IsoBDT + p2_IsoBDT
    track_Chi2Dof = np.sqrt(np.square(p0_track_Chi2Dof - 1.) + np.square(p1_track_Chi2Dof - 1.) + np.square(p2_track_Chi2Dof - 1.))
    IP_sum = p0_IP + p1_IP + p2_IP
    IPSig_sum = p0_IPSig + p1_IPSig + p2_IPSig
    CDF_sum = CDF1 + CDF2 + CDF3

    # Kinematic features related to the mother particle:
    df['E'] = E
    df['pz'] = pz
    df['beta'] = beta
    df['gamma'] = gamma
    df['beta_gamma'] = beta_gamma
    df['M_lt'] = M_lt
    df['M_inv'] = M_inv
    df['Delta_E'] = Delta_E
    df['Delta_M'] = Delta_M
    df['flag_M'] = flag_M

    # Kinematic features related to the final-state particles:
    df['E0'] = E0
    df['E1'] = E1
    df['E2'] = E2
    df['E0_ratio'] = E0_ratio
    df['E1_ratio'] = E1_ratio
    df['E2_ratio'] = E2_ratio
    df['p0_pt_ratio'] = p0_pt_ratio
    df['p1_pt_ratio'] = p1_pt_ratio
    df['p2_pt_ratio'] = p2_pt_ratio
    df['eta_01'] = eta_01
    df['eta_02'] = eta_02
    df['eta_12'] = eta_12
    df['t_coll'] = t_coll

    # Other extra features:
    df['FlightDistanceSig'] = FlightDistanceSig
    df['DOCA_sum'] = DOCA_sum
    df['isolation_sum'] = isolation_sum
    df['IsoBDT_sum'] = IsoBDT_sum
    df['track_Chi2Dof'] = track_Chi2Dof
    df['IP_sum'] = IP_sum
    df['IPSig_sum'] = IPSig_sum
    df['CDF_sum'] = CDF_sum

    return df


if __name__ == "__main__":
    # Prediction and output:
    df = pd.read_csv("tests/test_resources/flavors_of_physics_huge.csv", index_col='id')
    c = np.ones(2) * 299.792458
    # Add extra features:
    num_rows = len(df.index)
    mini_df = df.head(2).copy()
    add_features(mini_df)
    add_features(mini_df)
    add_features(mini_df)
    c = np.ones(len(df)) * 299.792458
    t0 = time.time()
    df = add_features(df)
    time_elapsed = time.time() - t0
    print("Featurization time: {}".format(time_elapsed))
    print("Featurization throughput (rows/sec): {}".format(num_rows / time_elapsed))
    print(df.values[0])
