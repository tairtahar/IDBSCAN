import numpy as np
from sklearn.neighbors import NearestNeighbors, KDTree
from random import sample
import os
from scipy.spatial.distance import pdist, squareform


class DensityGeneral:
    def __init__(self, eps, minpts, tau, save_flag, path, verbose):
        """L is a list that contains indices of the leaders in the data
        F is a list in the length of the data that contains the followers of each example
        For leader l, its followers exist in the list F[l]
        The elements that are not leader will have their list in F empty"""
        self.data = np.array([])
        self.m = 0
        self.dist_mat = []
        self.eps = eps
        self.minpts = minpts
        self.tau = tau
        self.L = []
        self.followers_not_leaders = []
        self.num_leaders = 0
        self.F = []
        self.F_parallel = []
        self.labels_ = []
        self.outliers = []
        self.tree = []
        self.leader_labels = []
        self.S_data = np.array([])   # that is subset data, the data of the leaders/leaders+addition for IDBSCAN
        self.S_idx = []
        self.neighbor_calc = 1  # default 1 which calculation with sklearn kdtree. otherwise use dist_map

        self.save_flag = save_flag
        self.path = path
        self.verbose = verbose

    def leader(self):
        self.L = [0]  # list of all idices of the leaders
        self.F_parallel = [[] for _ in range(len(self.data))]
        self.F_parallel[0] = [0]
        self.dist_mat = pdist(self.data)

        # list of lists, in the order of L, contains indices of the population represented by the same order element
        # of L
        for d_idx in range(self.m - 1):  # from index 1 till the end
            curr_idx = d_idx + 1
            leader = True
            for l_idx in range(len(self.L)):
                if self.find_specific_dist(self.L[l_idx], curr_idx) <= self.tau:
                    self.F_parallel[self.L[l_idx]].append(curr_idx)
                    leader = False
                    break
            if leader:
                self.L.append(curr_idx)
                self.F_parallel[curr_idx].append(curr_idx)
        if self.verbose:
            print("leader algo is Done")
            print("leaders found ", str(len(self.L)))
        self.num_leaders = len(self.L)
        self.validate_F_contains_all()

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
        idx_neighbors = np.array(self.S_idx)[idx_neighbors_S]
        return idx_neighbors

    def DBSCAN(self):
        cluster = 0
        self.leader_labels = [0] * (self.num_leaders)
        for d_idx in range(len(self.S_data)):
            if self.leader_labels[d_idx] == 0:
                # finds the indices of the samples in the radius eps around current
                self.tree = KDTree(self.S_data)
                NN = self.tree.query_radius(self.S_data[d_idx].reshape(1, -1), r=self.eps)[0]
                if NN.shape[0] < self.minpts:
                    self.leader_labels[d_idx] = -1  # labels as noise
                else:
                    cluster += 1
                    self.leader_labels[d_idx] = cluster
                    S = NN.copy()
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

    def passing_predictions(self):
        self.labels_ = [0] * self.m
        for idx_L in range(
                len(self.L)):  # that step would label each group of followers according to its leader prediction
            current_prediction = self.leader_labels[idx_L]
            current_leader_idx = self.L[idx_L]
            self.labels_[current_leader_idx] = current_prediction
            current_followers_idx = self.F_parallel[current_leader_idx]
            for follower_idx in current_followers_idx:
                self.labels_[follower_idx] = current_prediction
        if self.save_flag:
            with open(os.path.join(self.path, "labels.txt"), "w") as f:
                for label in self.labels_:
                    f.write(str(label) + "\n")
        if 0 in self.labels_:
            not_labeled = [i for i, e in enumerate(self.labels_) if e == 0]
            print(not_labeled)
            raise ValueError(str(len(not_labeled)), ' elements were not classified')

    def validate_F_contains_all(self):
        check_vec = [0] * self.m
        for i in range(self.m):
            for follower in self.F_parallel[i]:
                check_vec[follower] += 1
        if 0 in check_vec:
            not_claffified = [i for i, e in enumerate(check_vec) if e == 0]
            print(not_claffified)
            raise ValueError(str(len(not_claffified)), ' elements were not classified')
        elif self.verbose:
            print("all followers were assinged to at least one leader")

    def update_data(self, df):
        self.data = np.asarray(df)
        self.m = len(df)

    def fit(self, df):
        self.update_data(df)
        self.leader()
        self.S_data = self.data[self.L]
        self.DBSCAN()
        self.passing_predictions()
        return self


class IDBSCAN(DensityGeneral):
    def __init__(self, eps, minpts, tau, save_flag, path, flag_neig_calc=1, verbose=False):
        """L is a list that contains indices of the leaders in the data
        F is a list in the length of the data that contains the followers of each example
        For leader l, its follwers exist in the list F[l]
        The elements that are not leader will have their list in F empty"""
        self.data = np.array([])
        self.m = 0
        self.dist_mat = []
        self.eps = eps
        self.minpts = minpts
        self.tau = tau
        self.L = []
        self.num_leaders = 0
        self.F = []
        self.F_parallel = []
        self.labels_ = []
        self.outliers = []
        self.tree = []
        self.leader_labels = []
        self.save_flag = save_flag
        self.path = path
        self.S_idx = []
        self.S_data = np.array([])
        self.followers_not_leaders = []
        self.num_followers_not_leaders = 0
        self.neighbor_calc = flag_neig_calc
        self.verbose = verbose

    def leader_asterisk(self):
        self.L = [0]  # list of all idices of the leaders. Initialized with the first index
        self.F = [[] for _ in range(len(self.data))]
        self.F_parallel = [[] for _ in range(len(self.data))]
        self.F_parallel[0].append(0)
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
        if self.verbose:
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
                outliers.append(d_idx)
        if self.verbose:
            print("leader* complete")
        self.num_leaders = len(self.L)
        self.outliers = outliers
        self.validate_F_contains_all()

    def IDBSCAN_sampling(self):
        S = self.L.copy()
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
            if l_idx % 500 == 0 and self.verbose:
                print("IDBSCAN sample " + str(l_idx) + " out of " + str(self.num_leaders))
        self.num_followers_not_leaders = len(self.followers_not_leaders)
        return S

    def find_interesect_followers(self, l_idx):
        s = []
        l1 = self.L[l_idx]  # l1 is the index of the leader
        for l2 in self.L:
            if l2 == l1:
                continue
            intersection = list(set(self.F[l1]) & set(self.F[l2]))
            s.extend(intersection)
        s_final = list(set(s))
        return list(set(s_final))

    def FFT_sampling(self, s):
        """This function returns minpts examples (idx) that are far from each other"""
        fft_out = []
        s_copy = s.copy()
        current_idx = sample(range(len(s)), 1)[0]
        fft_out.append(s_copy[current_idx])
        while len(set(fft_out)) < self.minpts:
            row_distances = self.find_row(current_idx,
                                          s_copy)  # this isolate the row of the specific instance in the distance
            # matrix
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
        return fft_out

    def DBSCAN(self):
        cluster = 0
        self.leader_labels = [0] * (self.num_leaders + self.num_followers_not_leaders)
        for d_idx in range(len(self.S_data)):
            if self.leader_labels[d_idx] == 0:
                # finds the indices of the samples in the radius eps around current
                if self.neighbor_calc:
                    self.tree = KDTree(self.S_data)
                    NN = self.tree.query_radius(self.S_data[d_idx].reshape(1, -1), r=self.eps)[0]
                else:
                    NN = self.find_neighbors_in_radius(d_idx)
                # sample
                if NN.shape[0] < self.minpts:
                    self.leader_labels[d_idx] = -1  # labels as noise
                else:
                    cluster += 1
                    self.leader_labels[d_idx] = cluster
                    S = NN.copy()
                    S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                    addition = self.neighbors_labeling(S, d_idx, cluster)
                    while len(addition) > 0:
                        addition = self.neighbors_labeling(addition, d_idx, cluster)

    def fit(self, df):
        self.update_data(df)
        self.leader_asterisk()
        if self.verbose:
            print("leaders list contains", len(self.L))
        S = self.IDBSCAN_sampling()
        if self.verbose:
            print("Intersection followers list contains", len(self.followers_not_leaders))
            print("All samples to be processed list contains", len(S))
        # precautions and validations:
        if len(S) - self.num_followers_not_leaders != self.num_leaders:
            raise ValueError('S != sum length of leaders and intersections')
        # S contains the results of IDBSCAN - indices of the leaders (len = L) + indices of inersections (len=S-L)
        self.S_idx = S
        self.S_data = np.asarray(self.data[S])
        self.DBSCAN()
        predictions = self.leader_labels
        if len(predictions) != len(S):
            raise ValueError('prediction list contains', str(len(predictions)), 'while S list contains', str(len(S)))
        self.passing_predictions()
        return self

    def fit_predict(self, df):
        self.fit(df)
        return self.labels_


class DBSCAN_manual:
    def __init__(self, eps, minpts):
        self.eps = eps
        self.minpts = minpts
        self.data = np.array([])
        self.labels_ = []
        self.m = 0
        self.tree = []

    def update_data(self, df):
        self.data = np.asarray(df)
        self.m = len(df)

    def fit(self, D):
        self.update_data(np.asarray(D))
        cluster = 0
        self.labels_ = [0] * self.m
        for d_idx in range(self.m):
            if self.labels_[d_idx] == 0:
                self.tree = KDTree(self.data)
                NN = self.tree.query_radius(self.data[d_idx].reshape(1, -1), r=self.eps)[0]
                if NN.shape[0] < self.minpts:
                    self.labels_[d_idx] = -1  # labels as noise
                else:
                    cluster += 1
                    self.labels_[d_idx] = cluster
                    S = NN.copy()
                    S = np.setdiff1d(S, np.array(d_idx))  # get rid of the current index
                    addition = self.neighbors_labeling(S, d_idx, cluster)
                    while len(addition) > 0:
                        addition = self.neighbors_labeling(addition, d_idx, cluster)
        return self

    def neighbors_labeling(self, S, d_idx, cluster):
        addition_temp = []
        addition_out = []
        for q_idx in S:  # handle of the nearest neighbors
            if self.labels_[q_idx] == -1:  # in case it was labeled as noise - label as cluster
                self.labels_[q_idx] = cluster
            if self.labels_[q_idx] == 0:  # meaning label q is undefined
                self.labels_[q_idx] = cluster
                NN = self.tree.query_radius(self.data[q_idx].reshape(1, -1), r=self.eps)[0]
                if NN.shape[0] >= self.minpts:
                    addition_temp.append(NN)
        if len(addition_temp) > 0:
            addition_temp = np.concatenate(addition_temp).ravel().tolist()
            addition1 = [item for item in addition_temp if item not in S]  # all elements that do not exist already in S
            addition_out = np.setdiff1d(addition1, np.array(d_idx))  # get rid of the current index d_idx

        return addition_out

    def fit_predict(self, D):
        self.fit(D)
        return self.labels_

