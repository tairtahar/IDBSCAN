import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets as data
import hdbscan
import os
import utils
from sklearn.cluster import DBSCAN
import time
# os.chdir("/Users/tair/PycharmProjects")


df, true_class = utils.load_preprocess_abalone()
eps = 0.2
minpts = 3
# data_name == "abalone"  # 4,177 samples, works great
clustring = DBSCAN(eps=eps, min_samples=minpts).fit(np.asarray(df))  # sklearn
predictions_ref = clustring.labels_
print("baseline sklearn DBSCAN evaluation: ", )
utils.perform_evaluation(true_class, predictions_ref,
                   True)

start = time.time()
clusterer = hdbscan.HDBSCAN(min_samples=minpts, cluster_selection_epsilon=eps)
predictions = clusterer.fit(df).labels_
end = time.time()
time_elapsed = end - start
print("runtime: " + str(time_elapsed))
utils.perform_evaluation(predictions_ref, predictions,
                         True)
# import requests

# r = requests.get("http://google.com")
# print(r.status_code)

sns.set_context('poster')
sns.set_style('white')
sns.set_color_codes()
plot_kwds = {'alpha' : 0.5, 's' : 80, 'linewidths':0}


moons, _ = data.make_moons(n_samples=50, noise=0.05)
blobs, _ = data.make_blobs(n_samples=50, centers=[(-0.75,2.25), (1.0, 2.0)], cluster_std=0.25)
test_data = np.vstack([moons, blobs])
plt.scatter(test_data.T[0], test_data.T[1], color='b', **plot_kwds)
plt.show()

clusterer = hdbscan.HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clusterer.fit(test_data)

clusterer.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
plt.show()

clusterer.condensed_tree_.plot()
plt.show()

palette = sns.color_palette()

cluster_colors = [sns.desaturate(palette[col], sat)
                  if col >= 0 else (0.5, 0.5, 0.5) for col, sat in
                  zip(clusterer.labels_, clusterer.probabilities_)]
plt.scatter(test_data.T[0], test_data.T[1], c=cluster_colors, **plot_kwds)
plt.show()