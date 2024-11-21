from typing import TypedDict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN


class Vector(TypedDict):
    vector: list[float]
    id: str


def cluster(vectors: list[Vector], eps=45.0, min_samples=1) -> list[list[str]]:
    data = StandardScaler().fit_transform(
        np.array([vector["vector"] for vector in vectors])
    )
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean").fit(data)
    clusters: dict[str, list[str]] = {}
    for item, label in zip(vectors, db.labels_):
        clusters.setdefault(label, []).append(item["id"])
    formatted_clusters = [vectors for label, vectors in clusters.items() if label != -1]
    return formatted_clusters
