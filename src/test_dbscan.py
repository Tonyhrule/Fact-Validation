import numpy as np

from sklearn import metrics
from sklearn.cluster import DBSCAN

from helpers.data import read_json

data = read_json("data/pubmed_summarized.json")

print(data[0])

db = DBSCAN(eps=0.3, min_samples=10).fit(X)
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)
