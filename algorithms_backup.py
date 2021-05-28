import numpy as np
import utils
import random
from sklearn.neighbors import NearestNeighbors, KDTree
from random import sample
import pandas as pd
import os
from scipy.spatial.distance import pdist, squareform


# from sklearn.neighbors import BallTree, KDTree
class DensityGeneral:
    def __init__(self, data, eps, minpts, tau, save_flag, path):
        """L is a list that contains indices of the leaders in the data
        F is a list in the length of the data that contains the followers of each example
        For leader l, its follwers exist in the list F[l]
        The elements that are not leader will have their list in F empty"""
        self.data = data
        self.m = len(data)
        self.dist_mat = []
        self.eps = eps
        self.minpts = minpts
        self.tau = tau
        self.L = []
        self.followers_not_leaders = []
        self.num_leaders = 0
        self.F = []
        self.F_parallel = []
        self.labels = []
        self.outliers = []
        self.tree = []
        self.leader_labels = []
        self.save_flag = save_flag
        self.path = path
        self.S_data = []  # that is partial data, the data of the leaders/leaders+addition for IDBSCAN
        self.S_idx = []
        self.neighbor_calc = 1  # default 1 which calculation with sklearn kdtree. otherwise use dist_map

    def leader(self):
        self.L = [0]  # list of all idices of the leaders
        self.F = [[] for _ in range(len(self.data))]
        self.F[0] = [0]
        self.dist_mat = pdist(self.data)

        # list of lists, in the order of L, contains indices of the population represented by the same order element of L
        for d_idx in range(self.m - 1):  # from index 1 till the end
            # for d_idx in range(300):
            curr_idx = d_idx + 1
            leader = True
            for l_idx in range(len(self.L)):
                if self.find_specific_dist(self.L[l_idx], curr_idx) <= self.tau:
                    # if utils.l2norm(D[L[l_idx]], D[curr_idx]) <= tau:
                    self.F[self.L[l_idx]].append(curr_idx)
                    leader = False
                    break
            if leader:
                self.L.append(curr_idx)
                self.F[curr_idx].append(curr_idx)
        print("leader algo is Done")
        print("leaders found ", str(len(self.L)))
        self.num_leaders = len(self.L)
        self.validate_F_contains_all()
        # return L, F

    def neighbors_labeling(self, S, d_idx, cluster):
        addition_temp = []
        addition_out = []
        for q_idx in S:  # handle of the nearest neighbors
            if self.leader_labels[q_idx] == -1:  # in case it was labeled as noise - label as cluster
                self.leader_labels[q_idx] = cluster
            if self.leader_labels[q_idx] == 0:  # meaning label q is undefined
                self.leader_labels[q_idx] = cluster
                if self.neighbor_calc:
                    NN = self.tree.query_radius(self.S_data[q_idx].reshape(1, -1), r=self.eps)[0]
                else:
                    NN = self.find_neighbors_in_radius(q_idx)
                # idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, D[d_idx].size), return_distance=False)
                # NN = np.asarray(idx_NN[0])
                if NN.shape[0] >= self.minpts:
                    addition_temp.append(NN)
        if len(addition_temp) > 0:
            addition_temp = np.concatenate(addition_temp).ravel().tolist()
            addition1 = [item for item in addition_temp if item not in S]  # all elements that do not exist already in S
            addition_out = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx

        return addition_out

    def find_neighbors_in_radius(self, idx):
        row_distances = self.find_row(idx, self.S_idx)
        idx_neighbors_S = np.where(np.asarray(row_distances) <= self.eps)
        # temp_list = self.S_idx.copy()
        idx_neighbors = np.array(self.S_idx)[idx_neighbors_S]
        # [temp_list[item] for item in idx_neighbors_S]
        return idx_neighbors

    def DBSCAN(self):
        cluster = 0
        self.leader_labels = [0] * (self.num_leaders)
        for d_idx in range(len(self.S_data)):
            if self.leader_labels[d_idx] == 0:
                # NN = NearestNeighbors(D, d_idx, eps) neigh = NearestNeighbors(radius=eps) neigh.fit(D) idx_NN =
                # neigh.radius_neighbors(D[d_idx].reshape(1, D[d_idx].size), return_distance=False)  # finds the indices
                # of the samples in the radius eps around current
                self.tree = KDTree(self.S_data)
                NN = self.tree.query_radius(self.S_data[d_idx].reshape(1, -1), r=self.eps)[0]
                # sample
                # NN = np.asarray(idx_NN[0])
                if NN.shape[0] < self.minpts:
                    self.leader_labels[d_idx] = -1  # labels as noise
                else:
                    cluster += 1
                    self.leader_labels[d_idx] = cluster
                    S = NN.copy()
                    # S = S.astype(int)
                    S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                    addition = self.neighbors_labeling(S, d_idx, cluster)
                    while len(addition) > 0:
                        addition = self.neighbors_labeling(addition, d_idx, cluster)

    def find_specific_dist(self, idx1, idx2):
        if idx1 == idx2:
            return 0
        elif idx1 < idx2:
            distance = self.dist_mat[self.m * idx1 + idx2 - ((idx1 + 2) * (idx1 + 1)) // 2]
        else:
            distance = self.dist_mat[self.m * idx2 + idx1 - ((idx2 + 2) * (idx2 + 1)) // 2]
        return distance

    def find_row(self, idx, group_idx):  # extract the "row" of idx in the distance matrix
        row_dist = []
        for current_idx in group_idx:
            row_dist.append(self.find_specific_dist(group_idx[idx], current_idx))
        return row_dist

    def inverse_dist_to_idx(self, idx_vectorised):
        n = len(self.dist_mat)
        n_row_elems = np.cumsum(np.arange(1, n)[::-1])
        ii = (n_row_elems[:, None] - 1 < idx_vectorised[None, :]).sum(axis=0)
        shifts = np.concatenate([[0], n_row_elems])
        jj = np.arange(1, n)[ii] + idx_vectorised - shifts[ii]
        return ii, jj

    def passing_predictions(self, labels):
        for idx_L in range(
                len(self.L)):  # that step would label each group of followers according to its leader prediction
            current_prediction = self.leader_labels[idx_L]
            current_leader_idx = self.L[idx_L]
            labels[current_leader_idx] = current_prediction
            current_followers_idx = self.F_parallel[current_leader_idx]
            for follower_idx in current_followers_idx:
                labels[follower_idx] = current_prediction
        if self.save_flag:
            with open(os.path.join(self.path, "labels.txt"), "w") as f:
                for label in labels:
                    f.write(str(label) + "\n")
        if 0 in labels:
            not_labeled = [i for i, e in enumerate(labels) if e == 0]
            print(not_labeled)
            raise ValueError(str(len(not_labeled)), ' elements were not classified')
        self.labels = labels
        return labels

    def validate_F_contains_all(self):
        check_vec = [0] * self.m
        for i in range(self.m):
            for follower in self.F_parallel[i]:
                check_vec[follower] += 1
        if 0 in check_vec:
            not_claffified = [i for i, e in enumerate(check_vec) if e == 0]
            print(not_claffified)
            raise ValueError(str(len(not_claffified)), ' elements were not classified')
        else:
            print("all followers were assinged to at least one leader")


class DensityAsterisk(DensityGeneral):
    def __init__(self, data, eps, minpts, tau, save_flag, path, flag_neig_calc=1):
        """L is a list that contains indices of the leaders in the data
        F is a list in the length of the data that contains the followers of each example
        For leader l, its follwers exist in the list F[l]
        The elements that are not leader will have their list in F empty"""
        self.data = data
        self.m = len(data)
        self.dist_mat = []
        self.eps = eps
        self.minpts = minpts
        self.tau = tau
        self.L = []
        self.num_leaders = 0
        self.F = []
        self.F_parallel = []
        self.labels = [0] * self.m
        self.outliers = []
        self.tree = []
        self.leader_labels = []
        self.save_flag = save_flag
        self.path = path
        self.S_idx = []
        self.S_data = []
        self.followers_not_leaders = []
        self.num_followers_not_leaders = 0
        self.neighbor_calc = flag_neig_calc

    def leader_asterisk(self):
        self.L = [0]  # list of all idices of the leaders. Initialized with the first index
        self.F = [[] for _ in range(len(self.data))]
        self.F_parallel = [[] for _ in range(len(self.data))]
        self.F_parallel[0].append(0)
        # list of lists, in the order of L, contains indices of the population represented by the same order element of L
        self.dist_mat = pdist(self.data)
        for d_idx in range(self.m - 1):  # len(D[1:])
            curr_idx = d_idx + 1  # since we start with 1 insead of zero
            leader = True
            for l_idx in range(len(self.L)):
                curr_dist = self.find_specific_dist(curr_idx, self.L[l_idx])
                if curr_dist <= self.tau:
                    self.F_parallel[self.L[l_idx]].append(curr_idx)  # the original leader output
                    leader = False
                    break
            if leader:
                self.F_parallel[curr_idx].append(curr_idx)
                self.L.append(curr_idx)
        print("leader* first iteration on data Done")
        flag = False
        outliers = []
        for d_idx in range(self.m):
            for l_idx in self.L:
                curr_dist = self.find_specific_dist(d_idx, l_idx)
                if curr_dist <= self.eps:
                    self.F[l_idx].append(d_idx)
                    flag = True
            if not flag:
                print(d_idx)
                outliers.append(d_idx)
        print("leader* complete")
        self.num_leaders = len(self.L)
        self.outliers = outliers
        self.validate_F_contains_all()

    def IDBSCAN(self):
        S = self.L.copy()  # make sure this is a copy of the L list

        for l_idx in range(self.num_leaders):
            s = self.find_interesect_followers(l_idx)
            if len(self.F[self.L[l_idx]]) > self.minpts:
                if len(set(s)) > self.minpts:
                    s = self.FFT_sampling(s)
                else:
                    s.extend(sample(self.F[self.L[l_idx]], self.minpts - len(s)))
            clean_s = [item for item in s if item not in S]
            self.followers_not_leaders.extend(clean_s)
            S.extend(clean_s)
            if l_idx % 500 == 0:
                print("IDBSCAN sample " + str(l_idx) + " out of " + str(self.num_leaders))
        self.num_followers_not_leaders = len(self.followers_not_leaders)
        return S

    def neighbors_labeling(self, S, d_idx, cluster):
        addition_temp = []
        addition_out = []
        for q_idx in S:  # handle of the nearest neighbors
            if self.leader_labels[q_idx] == -1:  # in case it was labeled as noise - label as cluster
                self.leader_labels[q_idx] = cluster
            if self.leader_labels[q_idx] == 0:  # meaning label q is undefined
                self.leader_labels[q_idx] = cluster
                if self.neighbor_calc:
                    NN = self.tree.query_radius(self.S_data[q_idx].reshape(1, -1), r=self.eps)[0]
                else:
                    NN = self.find_neighbors_in_radius(q_idx)
                    NN = NN.reshape(1, -1)
                # idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, D[d_idx].size), return_distance=False)
                # NN = np.asarray(idx_NN[0])
                if NN.shape[0] >= self.minpts:
                    addition_temp.append(NN)
        if len(addition_temp) > 0:
            addition_temp = np.concatenate(addition_temp).ravel().tolist()
            addition1 = [item for item in addition_temp if item not in S]  # all elements that do not exist already in S
            addition_out = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx

        return addition_out

    def find_interesect_followers(self, l_idx):
        s = []
        l1 = self.L[l_idx]  # l1 is the index of the leader
        for l2 in self.L:
            if l2 == l1:
                continue
            intersection = list(set(self.F[l1]) & set(self.F[l2]))
            s.extend(intersection)
        s_final = list(set(s))
        # print("intersection calculation complete")
        return list(set(s_final))

    def FFT_sampling(self, s):
        """This function returns minpts examples (idx) that are far from each other"""
        fft_out = []
        s_copy = s.copy()
        current_idx = sample(range(len(s)), 1)[0]
        fft_out.append(s_copy[current_idx])
        while len(set(fft_out)) < self.minpts:
            row_distances = self.find_row(current_idx,
                                          s_copy)  # this isolate the row of the specific instance in the distance matrix
            if len(set(row_distances)) <= self.minpts - len(set(fft_out)):
                fft_out.extend(s_copy[0:self.minpts - len(set(fft_out))])
                return fft_out
            farthest_idx_s = np.argmax(row_distances)
            fft_out.append(s_copy[farthest_idx_s])
            del s_copy[current_idx]
            if farthest_idx_s >= current_idx:
                current_idx = farthest_idx_s - 1  # in case the deleted item is former, we need to -1 the latter items.
            else:
                current_idx = farthest_idx_s
            # dist_mat[current_idx][farthest_idx_s] = 0  # making sure we do not add this idx again
            # dist_mat[farthest_idx_s][current_idx] = 0
        return fft_out

    def DBSCAN(self):
        cluster = 0
        self.leader_labels = [0] * (self.num_leaders + self.num_followers_not_leaders)
        for d_idx in range(len(self.S_data)):
            if self.leader_labels[d_idx] == 0:
                # NN = NearestNeighbors(D, d_idx, eps) neigh = NearestNeighbors(radius=eps) neigh.fit(D) idx_NN =
                # neigh.radius_neighbors(D[d_idx].reshape(1, D[d_idx].size), return_distance=False)  # finds the indices
                # of the samples in the radius eps around current
                if self.neighbor_calc:
                    self.tree = KDTree(self.S_data)
                    NN = self.tree.query_radius(self.S_data[d_idx].reshape(1, -1), r=self.eps)[0]
                else:
                    NN = self.find_neighbors_in_radius(d_idx)
                # sample
                # NN = np.asarray(idx_NN[0])
                if NN.shape[0] < self.minpts:
                    self.leader_labels[d_idx] = -1  # labels as noise
                else:
                    cluster += 1
                    self.leader_labels[d_idx] = cluster
                    S = NN.copy()
                    # S = S.astype(int)
                    S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                    addition = self.neighbors_labeling(S, d_idx, cluster)
                    while len(addition) > 0:
                        addition = self.neighbors_labeling(addition, d_idx, cluster)

    def find_specific_dist(self, idx1, idx2):
        if idx1 == idx2:
            return 0
        elif idx1 < idx2:
            distance = self.dist_mat[self.m * idx1 + idx2 - ((idx1 + 2) * (idx1 + 1)) // 2]
        else:
            distance = self.dist_mat[self.m * idx2 + idx1 - ((idx2 + 2) * (idx2 + 1)) // 2]
        return distance

    def find_row(self, idx, group_idx):  # extract the "row" of idx in the distance matrix
        row_dist = []
        for current_idx in group_idx:
            row_dist.append(self.find_specific_dist(group_idx[idx], current_idx))
        return row_dist

    def inverse_dist_to_idx(self, idx_vectorised):
        n = len(self.dist_mat)
        n_row_elems = np.cumsum(np.arange(1, n)[::-1])
        ii = (n_row_elems[:, None] - 1 < idx_vectorised[None, :]).sum(axis=0)
        shifts = np.concatenate([[0], n_row_elems])
        jj = np.arange(1, n)[ii] + idx_vectorised - shifts[ii]
        return ii, jj

    def passing_predictions(self, labels):
        for idx_L in range(
                len(self.L)):  # that step would label each group of followers according to its leader prediction
            current_prediction = self.leader_labels[idx_L]
            current_leader_idx = self.L[idx_L]
            labels[current_leader_idx] = current_prediction
            current_followers_idx = self.F_parallel[current_leader_idx]
            for follower_idx in current_followers_idx:
                labels[follower_idx] = current_prediction
        if self.save_flag:
            with open(os.path.join(self.path, "labels.txt"), "w") as f:
                for label in labels:
                    f.write(str(label) + "\n")
        if 0 in labels:
            not_labeled = [i for i, e in enumerate(labels) if e == 0]
            print(not_labeled)
            raise ValueError(str(len(not_labeled)), ' elements were not classified')
        self.labels = labels
        return labels

    def validate_F_contains_all(self):
        check_vec = [0] * self.m
        for i in range(self.m):
            for follower in self.F_parallel[i]:
                check_vec[follower] += 1
        if 0 in check_vec:
            not_claffified = [i for i, e in enumerate(check_vec) if e == 0]
            print(not_claffified)
            raise ValueError(str(len(not_claffified)), ' elements were not classified')
        else:
            print("all followers were assinged to at least one leader")

    def find_neighbors_in_radius(self, idx):
        row_distances = self.find_row(idx, self.S_idx)
        idx_neighbors_S = np.where(np.asarray(row_distances) <= self.eps)
        # temp_list = self.S_idx.copy()
        idx_neighbors = np.array(self.S_idx)[idx_neighbors_S]
        # [temp_list[item] for item in idx_neighbors_S]
        return np.asarray(idx_neighbors_S)


def main_IDBSCAN(df, eps, minpts, tau, save_flag, path, flag_neig_calc):
    data = np.asarray(df)
    labels = [0] * len(data)
    algorithm = DensityAsterisk(data, eps, minpts, tau, save_flag, path, flag_neig_calc)

    if algorithm.save_flag:  # creating and loading
        # data should be ndarray
        algorithm.leader_asterisk()
        print("leaders list contains", len(algorithm.L))
        S = algorithm.IDBSCAN()
        print("Intersection followers list contains", len(algorithm.followers_not_leaders))
        print("All samples to be processed list contains", len(S))
        with open(os.path.join(path, "leaders_idx.txt"), "w") as f:
            for l in algorithm.L:
                f.write(str(l) + "\n")

        with open(os.path.join(path, "followers.txt"), "w") as f:
            for followers_list in algorithm.F:
                f.write(str(followers_list) + "\n")

        with open(os.path.join(path, "IDBSCAN_idx.txt"), "w") as f:
            for s in S:
                f.write(str(s) + "\n")

        with open(os.path.join(path, "intersection_idx.txt"), "w") as f:
            for follower in algorithm.followers_not_leaders:
                f.write(str(follower) + "\n")

        if len(S) - algorithm.num_followers_not_leaders != algorithm.num_leaders:
            raise ValueError('S != sum length of leaders and intersections')

    else:  # loading only
        with open(os.path.join(path, "leaders_idx.txt"), "r") as f:
            for line in f:
                algorithm.L.append(int(line.strip()))

        with open(os.path.join(path, "followers.txt"), "r") as f:
            for line in f:
                current_line = []
                if line != "[]\n":
                    for element in line.split(','):
                        current_line.append(int(element.strip(' []\n')))
                algorithm.F.append(current_line)

        with open(os.path.join(path, "IDBSCAN_idx.txt"), "r") as f:
            S = []
            for line in f:
                S.append(int(line.strip()))

        with open(os.path.join(path, "intersection_idx.txt"), "r") as f:
            for line in f:
                algorithm.followers_not_leaders.append(int(line.strip()))

        if len(S) - algorithm.num_followers_not_leaders != algorithm.num_leaders:
            raise ValueError('S != sum length of leaders and intersections')
    # S contains the results of IDBSCAN - indices of the leaders (len = L) + indices of inersections (len=S-L)
    algorithm.S_idx = S
    algorithm.S_data = np.asarray(algorithm.data[S])
    algorithm.DBSCAN()
    predictions = algorithm.leader_labels
    if len(predictions) != len(S):
        raise ValueError('prediction list contains', str(len(predictions)), 'while S list contains', str(len(S)))
    prediction_leaders = algorithm.leader_labels[
                         0:algorithm.num_leaders]  # the first in the list are the prediction of the leaders.
    labels = algorithm.passing_predictions(labels)
    algorithm.labels = labels
    return labels


def passing_predictions(L, F, prediction_leaders, labels, save_flag, path):
    for idx_L in range(
            len(L)):  # that step would label each group of followers according to its leader prediction
        current_prediction = prediction_leaders[idx_L]
        current_leader_idx = L[idx_L]
        labels[current_leader_idx] = current_prediction
        current_followers_idx = F[current_leader_idx]
        for follower_idx in current_followers_idx:
            labels[follower_idx] = current_prediction
    if save_flag:
        with open(os.path.join(path, "labels.txt"), "w") as f:
            for label in labels:
                f.write(str(label) + "\n")
    if 0 in labels:
        print([i for i, e in enumerate(labels) if e == 0])
        raise ValueError('some elements were not classified')
    return labels


class DensityLeaderOriginal(DensityGeneral):
    def __init__(self, data, eps, minpts, tau, save_flag, path):
        self.data = data
        self.m = len(data)
        self.dist_mat = []
        self.eps = eps
        self.minpts = minpts
        self.tau = tau
        self.L = []
        self.num_leaders = 0
        self.F = []
        self.labels = []
        self.outliers = []
        self.tree = []
        self.leader_labels = []
        self.save_flag = save_flag
        self.path = path
        self.S_data = []  # that is partial data, the data of the leaders/leaders+addition for IDBSCAN

    def leader(self):
        self.L = [0]  # list of all idices of the leaders
        self.F = [[] for _ in range(len(self.data))]
        self.F[0] = [0]
        self.dist_mat = pdist(self.data)

        # list of lists, in the order of L, contains indices of the population represented by the same order element of L
        for d_idx in range(self.m - 1):  # from index 1 till the end
            # for d_idx in range(300):
            curr_idx = d_idx + 1
            leader = True
            for l_idx in range(len(self.L)):
                if self.find_specific_dist(self.L[l_idx], curr_idx) <= self.tau:
                    # if utils.l2norm(D[L[l_idx]], D[curr_idx]) <= tau:
                    self.F[self.L[l_idx]].append(curr_idx)
                    leader = False
                    break
            if leader:
                self.L.append(curr_idx)
                self.F[curr_idx].append(curr_idx)
        print("leader algo is Done")
        print("leaders found ", str(len(self.L)))
        self.num_leaders = len(self.L)
        self.validate_F_contains_all()
        # return L, F


def find_specific_dist(dist_mat, m, idx1, idx2):
    if idx1 == idx2:
        return 0
    elif idx1 < idx2:
        distance = dist_mat[m * idx1 + idx2 - ((idx1 + 2) * (idx1 + 1)) // 2]
    else:
        distance = dist_mat[m * idx2 + idx1 - ((idx2 + 2) * (idx2 + 1)) // 2]
    return distance


def leader_asterisk(D, tau, eps):
    L = [0]  # list of all idices of the leaders. Initialized with the first index
    F = [[] for _ in range(len(D))]
    # list of lists, in the order of L, contains indices of the population represented by the same order element of L
    dist_mat = pdist(D)
    m = len(D)
    for d_idx in range(len(D[1:])):
        curr_idx = d_idx + 1  # since we start with 1 insead of zero
        leader = True

        for l_idx in range(len(L)):
            curr_dist = find_specific_dist(dist_mat, m, curr_idx, L[l_idx])
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
            curr_dist = find_specific_dist(dist_mat, m, d_idx, l_idx)
            if curr_dist <= eps:
                F[l_idx].append(d_idx)
                flag = True
        if flag == False:
            print(d_idx)
            outliers.append(d_idx)

    print("leader* complete")
    return L, F, outliers, dist_mat


def neighbors_labeling(S, d_idx, labels, cluster, tree, D, eps, minpts):
    addition_temp = []
    addition_out = []
    for q_idx in S:  # handle of the nearest neighbors
        if labels[q_idx] == -1:  # in case it was labeled as noise - label as cluster
            labels[q_idx] = cluster
        if labels[q_idx] == 0:  # meaning label q is undefined
            labels[q_idx] = cluster
            NN = tree.query_radius(D[q_idx].reshape(1, -1), r=eps)[0]

            # idx_NN = neigh.radius_neighbors(D[q_idx].reshape(1, D[d_idx].size), return_distance=False)
            # NN = np.asarray(idx_NN[0])
            if NN.shape[0] >= minpts:
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
            # NN = NearestNeighbors(D, d_idx, eps) neigh = NearestNeighbors(radius=eps) neigh.fit(D) idx_NN =
            # neigh.radius_neighbors(D[d_idx].reshape(1, D[d_idx].size), return_distance=False)  # finds the indices
            # of the samples in the radius eps around current
            tree = KDTree(D)
            NN = tree.query_radius(D[d_idx].reshape(1, -1), r=eps)[0]
            # sample
            # NN = np.asarray(idx_NN[0])
            if NN.shape[0] < minpts:
                labels[d_idx] = -1  # labels as noise
            else:
                cluster += 1
                labels[d_idx] = cluster
                S = NN.copy()
                # S = S.astype(int)
                S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                labels, addition = neighbors_labeling(S, d_idx, labels, cluster, tree, D, eps, minpts)
                while len(addition) > 0:
                    labels, addition = neighbors_labeling(addition, d_idx, labels, cluster, tree, D, eps, minpts)
    return labels


def passing_predictions(L, F, prediction_leaders, labels, save_flag, path):
    for idx_L in range(
            len(L)):  # that step would label each group of followers according to its leader prediction
        current_prediction = prediction_leaders[idx_L]
        current_leader_idx = L[idx_L]
        labels[current_leader_idx] = current_prediction
        current_followers_idx = F[current_leader_idx]
        for follower_idx in current_followers_idx:
            labels[follower_idx] = current_prediction
    if save_flag:
        with open(os.path.join(path, "labels.txt"), "w") as f:
            for label in labels:
                f.write(str(label) + "\n")
    if 0 in labels:
        print([i for i, e in enumerate(labels) if e == 0])
        raise ValueError('some elements were not classified')
    return labels
