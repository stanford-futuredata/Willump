# https://github.com/gramolin/flavours-of-physics

import numpy as np
import pandas as pd
import time
import willump.evaluation.willump_executor

# Physical constants:
c = 299.792458  # Speed of light
m_mu = 105.6583715  # Muon mass (in MeV)
m_tau = 1776.82  # Tau mass (in MeV)


# Function to add extra features:
# @willump.evaluation.willump_executor.willump_execute
def add_features(df):
    # Number of events:
    N = len(df)

    # Internal arrays:
    p012_p = np.zeros((3, N))
    p012_pt = np.zeros((3, N))
    p012_eta = np.zeros((3, N))
    p012_IsoBDT = np.zeros((3, N))
    p012_track_Chi2Dof = np.zeros((3, N))
    p012_IP = np.zeros((3, N))
    p012_IPSig = np.zeros((3, N))
    CDF123 = np.zeros((3, N))
    isolation = np.zeros((6, N))

    flag_M = np.zeros(N)

    # Read some of the original features:
    pt = df['pt'].values
    dira = df['dira'].values
    LifeTime = df['LifeTime'].values
    FlightDistance = df['FlightDistance'].values
    FlightDistanceError = df['FlightDistanceError'].values
    DOCAone = df['DOCAone'].values
    DOCAtwo = df['DOCAtwo'].values
    DOCAthree = df['DOCAthree'].values
    isolation[0] = df['isolationa'].values
    isolation[1] = df['isolationb'].values
    isolation[2] = df['isolationc'].values
    isolation[3] = df['isolationd'].values
    isolation[4] = df['isolatione'].values
    isolation[5] = df['isolationf'].values

    for j in range(3):
        p012_p[j] = df['p' + str(j) + '_p'].values
        p012_pt[j] = df['p' + str(j) + '_pt'].values
        p012_eta[j] = df['p' + str(j) + '_eta'].values
        p012_IsoBDT[j] = df['p' + str(j) + '_IsoBDT'].values
        p012_track_Chi2Dof[j] = df['p' + str(j) + '_track_Chi2Dof'].values
        p012_IP[j] = df['p' + str(j) + '_IP'].values
        p012_IPSig[j] = df['p' + str(j) + '_IPSig'].values
        CDF123[j] = df['CDF' + str(j + 1)].values

    # Differences between pseudorapidities of the final-state particles:
    eta_01 = p012_eta[0] - p012_eta[1]
    eta_02 = p012_eta[0] - p012_eta[2]
    eta_12 = p012_eta[1] - p012_eta[2]

    # Transverse collinearity of the final-state particles (equals to 1 if they are collinear):
    t_coll = sum(p012_pt) / pt

    # Longitudinal momenta of the final-state particles:
    p012_z = p012_pt * np.sinh(p012_eta)

    # Energies of the final-state particles:
    E012 = np.sqrt(np.square(m_mu) + np.square(p012_p))

    # Energy and momenta of the mother particle:
    E = sum(E012)
    pz = sum(p012_z)
    p = np.sqrt(np.square(pt) + np.square(pz))

    # Energies and momenta of the final-state particles relative to those of the mother particle:
    E012_ratio = E012 / E
    p012_pt_ratio = p012_pt / pt

    # Mass of the mother particle calculated from FlightDistance and LifeTime:
    beta_gamma = FlightDistance / (LifeTime * c)
    M_lt = p / beta_gamma

    # If M_lt is around the tau mass then flag_M = 1 (otherwise 0):
    for i in range(N):
        if np.fabs(M_lt[i] - m_tau - 1.44) < 17:
            flag_M[i] = 1

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
    isolation_sum = sum(isolation)
    IsoBDT_sum = sum(p012_IsoBDT)
    track_Chi2Dof = np.sqrt(sum(np.square(p012_track_Chi2Dof - 1.)))
    IP_sum = sum(p012_IP)
    IPSig_sum = sum(p012_IPSig)
    CDF_sum = sum(CDF123)

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
    df['E0'] = E012[0, :]
    df['E1'] = E012[1, :]
    df['E2'] = E012[2, :]
    df['E0_ratio'] = E012_ratio[0, :]
    df['E1_ratio'] = E012_ratio[1, :]
    df['E2_ratio'] = E012_ratio[2, :]
    df['p0_pt_ratio'] = p012_pt_ratio[0, :]
    df['p1_pt_ratio'] = p012_pt_ratio[1, :]
    df['p2_pt_ratio'] = p012_pt_ratio[2, :]
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
    df = pd.read_csv("tests/test_resources/flavors_of_physics_test.csv", index_col='id')
    # Add extra features:
    num_rows = len(df.index)
    t0 = time.time()
    mini_df = df.head(2).copy()
    add_features(mini_df)
    add_features(mini_df)
    add_features(mini_df)
    df = add_features(df)
    time_elapsed = time.time() - t0
    print("Featurization time: {}".format(time_elapsed))
    print("Featurization throughput (rows/sec): {}".format(num_rows / time_elapsed))
    print(df.values[0])
