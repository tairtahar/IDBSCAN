import utils
from algorithms import DBSCAN_manual, IDBSCAN, DensityGeneral
import numpy as np
from sklearn.cluster import DBSCAN
import time
import hdbscan
from st_dbscan import ST_DBSCAN


def run_main(algos, data_name, flag_calc_neig, flag_save, path, verbose_IDBSCAN):
    print("dataset chosen is ", data_name)
    if data_name == "abalone":  # 4,177 samples
        df, true_class = utils.load_preprocess_abalone()
        eps = 0.2
        minpts = 3

    elif data_name == "mushroom":  # 8,124 samples
        df, true_class = utils.load_preprocess_mushrooms()
        eps = 2.5
        minpts = 4

    elif data_name == "pendigit":  # 10,992 samples
        df, true_class = utils.load_preprocess_pendigit()
        eps = 40
        minpts = 4

    elif data_name == "letter":  # 20,000 samples
        df, true_class = utils.load_preprocess_letters()
        eps = 0.5
        minpts = 8

    elif data_name == "cadata":  # 20,000 samples
        df, true_class = utils.load_preprocess_catadata()
        eps = 200
        minpts = 8

    elif data_name == "sensorless":  # 58,509 samples
        df, true_class = utils.load_preprocess_sensorless()
        eps = 0.3
        minpts = 20

    elif data_name == "shuttle":  # 58,000 samples
        df, true_class = utils.load_preprocess_shuttle()
        eps = 0.03
        minpts = 20

    tau = eps  # for IDBSCAN tau equals epsilon

    clustring = DBSCAN(eps=eps, min_samples=minpts).fit(np.asarray(df))  # sklearn
    predictions_ref = clustring.labels_
    print("baseline sklearn DBSCAN evaluation: ", )
    utils.perform_evaluation(true_class, predictions_ref, True)
    for i in range(len(algos)):
        algo = algos[i]
        start = time.time()

        if algo == "IDBSCAN":
            utils.print_separator()
            print("For my IDBSCAN:")
            clustering = IDBSCAN(eps, minpts, tau, flag_save, path, flag_calc_neig,
                                 verbose_IDBSCAN).fit(df)
            predictions = clustering.labels_
            # predictions = IDBSCAN(eps, minpts, tau, flag_save, path, flag_calc_neig, verbose_IDBSCAN).fit_predict(df)

        elif algo == "DBSCAN":
            utils.print_separator()
            print("For my DBSCAN:")
            clustering = DBSCAN_manual(eps, minpts).fit(df)
            predictions = clustering.labels_
            # predictions = DBSCAN_manual(eps, minpts).fit_predict(df)

        elif algo == "hdbscan":
            utils.print_separator()
            print("For HDBSCAN:")
            clusterer = hdbscan.HDBSCAN(min_samples=minpts, cluster_selection_epsilon=eps)
            predictions = clusterer.fit(df).labels_

        elif algo == "stdbscan":
            utils.print_separator()
            print("For ST-DBSCAN:")
            st_dbscan = ST_DBSCAN(eps1=eps, eps2=eps, min_samples=minpts)
            # st_dbscan.fit_frame_split(df, frame_size=100)
            st_dbscan.fit(df)
            predictions = st_dbscan.labels

        elif algo == "leader":
            utils.print_separator()
            print("For my Leader:")
            clustering = DensityGeneral(eps, minpts, tau, flag_save, path, verbose).fit(np.asarray(df))
            predictions = clustering.labels_

        end = time.time()
        time_elapsed = end - start
        print("runtime: " + str(time_elapsed))
        utils.perform_evaluation(predictions_ref, predictions, True)


algos = ["IDBSCAN", "DBSCAN", "stdbscan", "hdbscan"]  # , "leader" # "vdbscan",
datasets = ["abalone", "mushroom", "pendigit", "letter", "cadata", "sensorless", "shuttle"]
data_num = 0
flag_calc_neig = 1  # 1 uses sklearn KDtree and 0 uses distance matrix calculated by leader* alg (pdist).
flag_save = 1  # to save the labels outputs
path = "Results"  # where to save in case flag_save==1. Make sure the path exists.
verbose = False
data_name = datasets[data_num]

run_main(algos, data_name, flag_calc_neig, flag_save, path, verbose)
