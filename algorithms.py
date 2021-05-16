import numpy as np
import utils
import random
from sklearn.neighbors import NearestNeighbors, KDTree
from random import sample
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform


def leader(D, tau):
    L = [0]  # list of all idices of the leaders
    F = [[] for _ in range(len(D))]
    F[0] = [0]
    # list of lists, in the order of L, contains indices of the population represented by the same order element of L
    for d_idx in range(len(D[1:])):
        # for d_idx in range(300):
        curr_idx = d_idx + 1
        leader = True
        for l_idx in range(len(L)):
            if utils.l2norm(D[L[l_idx]], D[curr_idx]) <= tau:
                F[l_idx].append(curr_idx)
                leader = False
                break
        if leader:
            L.append(curr_idx)
            F[curr_idx].append(curr_idx)
    print("leader algo is Done")
    return L, F


def leader_asterisk(D, tau, eps):
    L = [0]  # list of all idices of the leaders. Initialized with the first index
    F = [[] for _ in range(len(D))]
    # list of lists, in the order of L, contains indices of the population represented by the same order element of L
    dist_mat = pdist(D)
    m = len(D)
    for d_idx in range(len(D[1:])):
        # for d_idx in range(300): # FOR DEBUG ONLY
        curr_idx = d_idx + 1  #since we start with 1 insead of zero
        leader = True

        for l_idx in range(len(L)):
            if curr_idx < L[l_idx]:
                curr_dist = dist_mat[m*curr_idx+L[l_idx]-((curr_idx+2)*(curr_idx+1))//2]
            else:
                curr_dist = dist_mat[m*L[l_idx]+curr_idx-((L[l_idx]+2)*(L[l_idx]+1))//2]
            if curr_dist <= tau:
                leader = False
                break
        if leader:
            L.append(curr_idx)
    print("leader* first iteration on data Done")
    flag = False
    outliers = []
    for d_idx in range(len(D)):
        for l_idx in L:
            if d_idx < l_idx:
                curr_dist = dist_mat[m*d_idx+l_idx-((d_idx+2)*(d_idx+1))//2]
            else:
                curr_dist = dist_mat[m*l_idx+d_idx-((l_idx+2)*(l_idx+1))//2]
            if curr_dist <= eps:
                F[l_idx].append(d_idx)
                flag = True
        if flag == False:
            print(d_idx)
            outliers.append(d_idx)

    print("leader* complete")
    return L, F, outliers


def find_interesect_followers(l_idx, L: list, F: list):
    s = []
    l1 = L[l_idx]  # l1 is the index of the leader
    for l2 in L:
        if l2 == l1:
            continue
        intersection = list(set(F[l1]) & set(F[l2]))
        s.extend(intersection)
    s_final = list(set(s))
    # print("intersection calculation complete")
    return s_final


def FFT_sampling(D, s, minpts):
    """This function returns minpts examples (idx) that are far from each other"""
    fft_out = []
    current_idx = sample(range(len(s)), 1)[0]
    fft_out.append(s[current_idx])
    dist_mat = squareform(pdist(D[s]))
    while len(list(set(fft_out))) < minpts:
        farthest_idx_s = np.argmax(dist_mat[current_idx])
        fft_out.append(s[farthest_idx_s])
        dist_mat[current_idx][farthest_idx_s] = 0  # making sure we do not add this idx again
        dist_mat[farthest_idx_s][current_idx] = 0
        current_idx = farthest_idx_s

    return list(set(fft_out))


def IDBSCAN(data, L, F, minpts):
    S = L.copy()  # make sure this is a copy of the L list
    followers_not_leaders = []
    for l_idx in range(len(L)):
        s = find_interesect_followers(l_idx, L, F)
        if len(F[L[l_idx]]) > minpts:
            if len(s) > minpts:
                s = FFT_sampling(data, s, minpts)
            else:
                s.extend(sample(F[L[l_idx]], minpts - len(s)))
        clean_s = [item for item in s if item not in S]
        followers_not_leaders.extend(clean_s)
        S.extend(clean_s)
        if l_idx % 500 == 0:
            print("IDBSCAN sample " + str(l_idx) + " out of " + str(len(L)))
    # flat_S = [idx_S for sublist in S for idx_S in sublist]
    # S is the idx of the leaders and appendices
    return S, followers_not_leaders


def neighboors_labeling(S, d_idx, labels, cluster, neigh, D, minpts):
    addition_temp = []
    addition_out = []
    for q_idx in S:  # handle of the nearest neighboors
        if labels[q_idx] == -1:  # labeled as noise
            labels[q_idx] = cluster
        if labels[q_idx] == 0:  # meaning label q is undefined
            labels[q_idx] = cluster
            idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, D[d_idx].size), return_distance=False)
            NN = np.asarray(idx_NN[0])
            if len(NN) >= minpts:
                addition_temp.append(NN)
    if len(addition_temp) > 0:
        addition_temp = np.concatenate(addition_temp).ravel().tolist()
        addition1 = [item for item in addition_temp if item not in S]  # all elements that do not exist already in S
        addition_out = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx

    return labels, addition_out


def DBSCAN(D, eps, minpts):
    cluster = 0
    labels = [0] * len(D)
    for d_idx in range(len(D)):
        if labels[d_idx] == 0:
            # NN = NearestNeighbors(D, d_idx, eps)
            neigh = NearestNeighbors(radius=eps)
            neigh.fit(D)
            idx_NN = neigh.radius_neighbors(D[d_idx].reshape(1, D[d_idx].size),
                                            return_distance=False)  # finds the indices of the samples in the radius eps around current
            # sample
            NN = np.asarray(idx_NN[0])
            if len(NN) < minpts:
                labels[d_idx] = -1  # labels as noise
            else:
                cluster += 1
                labels[d_idx] = cluster
                S = NN.copy()
                # S = S.astype(int)
                S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                labels, addition = neighboors_labeling(S, d_idx, labels, cluster, neigh, D, minpts)
                while len(addition) > 0:
                    labels, addition = neighboors_labeling(addition, d_idx, labels, cluster, neigh, D, minpts)
    return labels


"""L is a list that contains indices of the leaders in the data
F is a list in the length of the data that contains the followers of each example
For leader l, its follwers exist in the list F[l]
The elements that are not leader will have their list in F empty"""


def main_IDBSCAN(df, eps, minpts, save_flag, path):
    data = np.asarray(df)
    labels = [0] * len(data)

    if save_flag:  #creating and loading
        # data should be ndarray
        L, F, outliers = leader_asterisk(data, eps, eps)
        print("leaders list contains", len(L))
        S, followers_not_leaders = IDBSCAN(data, L, F, minpts)
        print("Intersection followers list contains", len(followers_not_leaders))
        print("All samples to be processed list contains", len(S))
        with open(os.path.join(path, "leaders_idx.txt"), "w") as f:
            for l in L:
                f.write(str(l) + "\n")

        with open(os.path.join(path, "followers.txt"), "w") as f:
            for followers_list in F:
                f.write(str(followers_list) + "\n")

        with open(os.path.join(path, "IDBSCAN_idx.txt"), "w") as f:
            for s in S:
                f.write(str(s) + "\n")

        with open(os.path.join(path, "intersection_idx.txt"), "w") as f:
            for follower in followers_not_leaders:
                f.write(str(follower) + "\n")

        if len(S)-len(followers_not_leaders) != len(L):
            raise ValueError('S != sum length of leaders and intersections')

    else:  # loading only
        with open(os.path.join(path, "leaders_idx.txt"), "r") as f:
            L = []
            for line in f:
                L.append(int(line.strip()))

        with open(os.path.join(path, "followers.txt"), "r") as f:
            F = []
            for line in f:
                current_line = []
                if line != "[]\n":
                    for element in line.split(','):
                        current_line.append(int(element.strip(' []\n')))
                F.append(current_line)

        with open(os.path.join(path, "IDBSCAN_idx.txt"), "r") as f:
            S = []
            for line in f:
                S.append(int(line.strip()))

        with open(os.path.join(path, "intersection_idx.txt"), "r") as f:
            followers_not_leaders = []
            for line in f:
                followers_not_leaders.append(int(line.strip()))

        if len(S)-len(followers_not_leaders) != len(L):
            raise ValueError('S != sum length of leaders and intersections')
    # S contains the results of IDBSCAN - indices of the leaders (len = L) + indices of inersections (len=S-L)
    prediction = DBSCAN(np.asarray(df.loc[S]), eps, minpts)
    if len(prediction) != len(S):
        raise ValueError('prediction list contains', len(prediction), 'while S list contains', len(S))
    prediction_leaders = prediction[0:len(L)]  # the first in the list are the prediction of the leaders.
    for idx_L in range(len(L)): #that step would label each group of followers according to its leader prediction
        current_prediction = prediction_leaders[idx_L]
        current_leader_idx = L[idx_L]
        labels[current_leader_idx] = current_prediction
        current_followers_idx = F[current_leader_idx]
        for follower_idx in current_followers_idx:
            labels[follower_idx] = current_prediction
    if 0 in labels:
        print([i for i, e in enumerate(labels) if e == 0])
        raise ValueError('some elements were not classified')

    return labels
