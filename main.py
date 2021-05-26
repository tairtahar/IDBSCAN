import utils
import algorithms
import numpy as np
from sklearn.cluster import DBSCAN
import time


algos = ["IDBSCAN", "DBSCAN"]
# algos = ["leader"]
data_name = "abalone"
print("dataset chosen is ", data_name)
if data_name == "mushroom":  # 8,124 samples, working
    df, true_class = utils.load_preprocess_mushrooms()  # one hot
    eps = 2.5
    minpts = 4
elif data_name == "letter":  # 20,000 samples
    df, true_class = utils.load_preprocess_letters()
    eps = 0.5
    minpts = 8
elif data_name == "pendigit":  # 10,992 samples, 0.739 ARI, and IDBSCAN takes longer
    df, true_class = utils.load_preprocess_pendigit()
    eps = 40
    minpts = 4
elif data_name == "abalone":  # 4,177 samples, works great
    df, true_class = utils.load_preprocess_abalone()
    eps = 0.2
    minpts = 3
elif data_name == "sensorless":  # 58,509 samples
    df, true_class = utils.load_preprocess_sensorless()
    eps = 0.3
    minpts = 20
elif data_name == "cadata":  # 20,000 woorks but slowly
    df, true_class = utils.load_preprocess_catadata()
    eps = 200
    minpts = 8
elif data_name == "shuttle":
    df, true_class = utils.load_preprocess_shuttle()
    eps = 0.03
    minpts = 20
elif data_name == "skin_nonskin":
    df, true_class = utils.load_preprocess_skin_nonskin()
    eps = 60
    minpts = 10
# elif data_name == "seismic":
#     df, true_class = utils.load_preprocess_seismic()
#     eps = 0.4
#     minpts = 5
tau = eps  # for IDBSCAN tau equals epsilon

clustring = DBSCAN(eps=eps, min_samples=minpts).fit(np.asarray(df))  # sklearn
predictions_ref = clustring.labels_
print("baseline sklearn DBSCAN evaluation: ", )
utils.perform_evaluation(true_class, predictions_ref,
                   True)
for i in range(len(algos)):
    algo = algos[i]
    start = time.time()
    if algo == "IDBSCAN":
        predictions = algorithms.main_IDBSCAN(df, eps, minpts, tau, True, "Results/results_abalone")
        print("For my IDBSCAN:")
    elif algo == "DBSCAN":
        predictions = algorithms.DBSCAN(np.asarray(df), eps, minpts)
        print("For my DBSCAN:")
    elif algo == "leader":
        leader_dbscan = algorithms.Density(df, eps, minpts, tau)
        L, F = leader_dbscan.leader(df, tau)
        predictions_leaders = algorithms.DBSCAN(df.loc[L], eps, minpts)
        labels = [0]*len(df)
        predictions = algorithms.passing_predictions(L, F, predictions_leaders, labels, True, "leader_algo.txt")
    end = time.time()
    time_elapsed = end - start
    print("runtime: " + str(time_elapsed))
    # perform_evaluation(data, true_class, predictions, True)  #make sure the data here should be the original without one hot
    utils.perform_evaluation(predictions_ref, predictions,
                       True)  # make sure the data here should be the original without one hot


