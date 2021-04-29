import utils
import algorithms
import numpy as np
import evaluation


def perform_evaluation(data, true_class, predictions, verbose=False):
    ARI = evaluation.ARI(true_class, predictions)
    silhouette_coefficient = evaluation.silhouette_coefficient(data, predictions)
    calinsky_harabasz = evaluation.calinsky_harabasz_index(data, predictions)
    davies_boulding = evaluation.davies_boulding_index(data, predictions)
    if verbose:
        print("ARI: " + str(ARI))
        print("Silhouette: " + str(silhouette_coefficient))
        print("calinsky harabasz index: " + str(calinsky_harabasz))
        print("davies boulding index: " + str(davies_boulding))

    return ARI, silhouette_coefficient, calinsky_harabasz, davies_boulding


# if __name__ == 'main':
data = utils.load_data('mushroom_arff.arff')
# data.info()
df = utils.categorial_handle(data, 2)
true_class = df.iloc[:, -1]
df = df.drop(df.columns[-1], axis=1)
eps = 2.5
tau = eps
minpts = 4
predictions = algorithms.main_IDBSCAN(df, eps, minpts)
perform_evaluation(data, true_class, predictions)

## uncomment the following lines for a full execution
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
