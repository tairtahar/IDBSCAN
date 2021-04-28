import numpy as np
import utils
import random
from sklearn.neighbors import NearestNeighbors, KDTree
from random import sample


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
    L = [0]  # list of all idices of the leaders
    F = [[] for _ in range(len(D))]
    # list of lists, in the order of L, contains indices of the population represented by the same order element of L
    for d_idx in range(len(D[1:])):
        # for d_idx in range(300): # FOR DEBUG ONLY
        curr_idx = d_idx + 1
        leader = True
        for l_idx in range(len(L)):
            if utils.l2norm(D[L[l_idx]], D[curr_idx]) <= tau:
                leader = False
                break
        if leader:
            L.append(curr_idx)
    print("leader* first iteration on data Done")
    for d_idx in range(len(D)):
        for l_idx in range(len(L)):
            if utils.l2norm(D[L[l_idx]], D[d_idx]) <= eps:
                F[l_idx].append(d_idx)

    print("leader* complete")
    return L, F


def find_interesect_followers(l_idx, L: list, F: list):
    s = []
    l1 = L[l_idx]
    for l2 in L:
        if l2 == l1:
            continue
        intersection = list(set(F[l1]) & set(F[l2]))
        s.extend(intersection)
    # flat_s = [idx_s for sublist in s for idx_s in sublist]
    s_final = list(set(s))
    # print("intersection calculation complete")
    return s_final


def FFT_sampling(D, s, minpts):
    """This function returns minpts examples (idx) that are far from each other"""
    fft_out = []
    current_idx = sample(range(len(s)), 1)[0]
    fft_out.append(s[current_idx])
    while len(fft_out) < minpts:
        longest_dist = 0
        for j in range(len(s)):
            if current_idx == j or s[j] in fft_out:
                continue
            if utils.l2norm(D[s[current_idx]], D[s[j]]) > longest_dist:
                longest_dist = utils.l2norm(D[s[current_idx]], D[s[j]])
                farest_idx = j
        current_idx = farest_idx
        fft_out.append(current_idx)
    return list(set(fft_out))


def neighboors_labeling(S, d_idx, labels, cluster, neigh, D, minpts):
    addition_temp = []
    addition_out = []
    for q_idx in S:  # handle of the nearest neighboors
        if labels[q_idx] == -1:  # labeled as noise
            labels[q_idx] = cluster
        if labels[q_idx] == 0:  # meaning label q is undefined
            labels[q_idx] = cluster
            idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, 23), return_distance=False)
            NN = np.asarray(idx_NN[0])
            if len(NN) >= minpts:
                addition_temp.append(NN)
    if len(addition_temp) > 0:
        addition_temp = np.concatenate(addition_temp).ravel().tolist()
        addition1 = [item for item in addition_temp if item not in S]  # all elements that do not exist already in S
        addition_out = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx
    # addition_out.extend(addition2)

    return labels, addition_out


def IDBSCAN(data, eps, minpts):
    L, F = leader_asterisk(data, eps, eps)
    S = L.copy()  # make sure this is a copy of the L list
    for l_idx in range(len(L)):
        s = find_interesect_followers(l_idx, L, F)
        if len(F[L[l_idx]]) > minpts:
            if len(s) > minpts:
                s = FFT_sampling(data, s, minpts)
            else:
                s.extend(sample(F[L[l_idx]], minpts - len(s)))
        clean_s = [item for item in s if item not in S]
        S.extend(clean_s)
    if l_idx % 500 == 0:
        print("IDBSCAN sample " + str(l_idx) + " out of " + str(len(data)))
    # flat_S = [idx_S for sublist in S for idx_S in sublist]
    # S is the idx of the leaders and appendices
    return S


def DBSCAN(D, eps, minpts):
    cluster = 0
    labels = [0] * len(D)
    for d_idx in range(len(D)):
        if labels[d_idx] == 0:
            # NN = NearestNeighbors(D, d_idx, eps)
            neigh = NearestNeighbors(radius=eps)
            neigh.fit(D)
            idx_NN = neigh.radius_neighbors(D[d_idx].reshape(1, 23),
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
                # for q_idx in S:  # handle of the nearest neighboors
                #     if labels[q_idx] == -1:  # labeled as noise
                #         labels[q_idx] = cluster
                #     if labels[q_idx] == 0:  # meaning label q is undefined
                #         labels[q_idx] = cluster
                #         idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, 23), return_distance=False)
                #         NN = np.asarray(idx_NN[0])
                #         if len(NN) >= minpts:
                #             addition1 = [item for item in NN if
                #                          item not in S]  # all elements that do not exist already in S
                #             addition2 = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx
                #             if len(addition2) > 0:
                #                 S = np.append(S, addition2)
    return labels


"""L is a list that contains indices of the leaders in the data
F is a list in the length of the data that contains the followers of each example
For leader l, its follwers exist in the list F[l]
The elements that are not leader will have their list in F empty"""

# FOR DEBUG ONLY
