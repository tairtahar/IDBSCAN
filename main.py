import utils
import algorithms
import numpy as np
from sklearn.cluster import DBSCAN
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd





algos = ["IDBSCAN", "DBSCAN"]
data_name = "abalone"
print("dataset chosen is ", data_name)
if data_name == "mushroom":  # 8,124 samples, working
    df, true_class = utils.load_preprocess_mushrooms()  # one hot
    eps = 2.5
    minpts = 4
    # eps = 3.8
    # minpts = 680

elif data_name == "letter":  # 20,000 samples
    df, true_class = utils.load_preprocess_letters()
    eps = 0.5
    minpts = 8
# data.info()
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
elif data_name == "seismic":
    df, true_class = utils.load_preprocess_seismic()
    eps = 0.4
    minpts = 5
tau = eps

clustring = DBSCAN(eps=eps, min_samples=minpts).fit(np.asarray(df))
predictions_ref = clustring.labels_
print("baseline sklearn DBSCAN evaluation: ", )
utils.perform_evaluation(true_class, predictions_ref,
                   True)
for i in range(len(algos)):
    algo = algos[i]
    start = time.time()
    if algo == "IDBSCAN":
        predictions = algorithms.main_IDBSCAN(df, eps, minpts, True, "Results/results_abalone")
        print("For my IDBSCAN:")

    elif algo == "DBSCAN":
        predictions = algorithms.DBSCAN(np.asarray(df), eps, minpts)
        print("For my DBSCAN:")
    end = time.time()
    time_elapsed = end - start
    print("runtime: " + str(time_elapsed))
    # perform_evaluation(data, true_class, predictions, True)  #make sure the data here should be the original without one hot
    utils.perform_evaluation(predictions_ref, predictions,
                       True)  # make sure the data here should be the original without one hot

# ## uncomment the following lines for a full execution
# S, followers_interserc = algorithms.IDBSCAN(np.asarray(df), eps, minpts)
# with open("IDBSCAN_idx.txt", "w") as f:
#     for s in S:
#         f.write(str(s) +"\n")
#
# with open("intersection_idx.txt", "w") as f:
#     for follower in followers_interserc:
#         f.write(str(follower) +"\n")


# for debug purposes - loading previously saved leader_idx's
# leaders_df = pd.DataFrame()
# with open("IDBSCAN_idx.txt", "r") as f:
#     S = []
#     for line in f:
#         S.append(int(line.strip()))
#
# print("length original data = " + str(len(np.asarray(df))))
# print("length of S after IDBSCAN = " + str(len(S)))
# prediction = algorithms.DBSCAN(np.asarray(df.loc[S]), eps, minpts)
#
# # FOR DEBUG ONLY
# print(len(prediction))
# print(prediction[0:41])
# print(prediction[1696])
# print(prediction[1741])
# print('=====')
# print(prediction[1703])
# print(prediction[1697])
# print(prediction[9])
# print(prediction[243])

# print("max = " + str(max(S)))
# print("min = " + str(min(S)))

# L, F = algorithms.leader_asterisk(df, tau, eps)
# L_idx, F_idx_lst = algorithms.leader_asterisk(np.asarray(df), tau, eps)
# data = np.asarray(df)
# print(L_idx[0:20])
# print(len(L_idx))
# # print(len(F_idx_lst))
# for i in range(20):
#     print("L index = " + str(L_idx[i]) + "; vals= " + str(data[L_idx[i]]))
#     print("followers = " + str(F_idx_lst[L_idx[i]]))
#     # print(F_idx_lst[0])

# print(utils.l2norm(data[L_idx[0]],data[19]))
# print(utils.l2norm(data[L_idx[0]],data[2]))
# print(utils.l2norm(data[L_idx[2]],data[22]))
# print(utils.l2norm(data[L_idx[2]],data[5]))

# for item in F_idx_lst:
#     if len(item) >= 2:
#         print(item)

# # algorithms.IDBSCAN(data, 2.5, 4)
