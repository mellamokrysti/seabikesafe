from inspect import CO_VARKEYWORDS
import numpy as np, pandas as pd, matplotlib.pyplot as plt, time
from geopandas import GeoDataFrame
from geopy.distance import great_circle
from openpyxl.workbook import Workbook
from pyproj import CRS
from shapely.geometry import MultiPoint, Point
from sklearn import cluster, metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

collisions_file = r'C:\SeaBikeSafe\seabikesafe\Collisions.csv'
df = pd.read_csv(collisions_file, dtype={"X": float, "Y": float, "PEDCYLCOUNT":int, "SEVERITYCODE":"string"}, low_memory=False, index_col=False)
df = df[(df['PEDCYLCOUNT'] != 0) & ((df["SEVERITYCODE"] == "2" ) | (df["SEVERITYCODE"] == "2b") | (df["SEVERITYCODE"] == "3"))]
df_locations = df[['X', 'Y']]
df_locations = df_locations.dropna()
geometry = [Point(xy) for xy in zip(df_locations.X, df_locations.Y)]
#df_locations = df_locations.drop(['X','Y'], axis=1)
#geo = GeoDataFrame(df_locations, crs="EPSG:4326", geometry=geometry)
epsilon = 0.075
#print(geo.columns)
#coords = geo[['x', 'y']].values
coords = StandardScaler().fit_transform(df_locations)
db = (DBSCAN(eps=epsilon, min_samples=100, algorithm='ball_tree', metric = 'euclidean').fit(coords))


# # ELBOW METHOD
# # determining the maximum number of clusters 
# # using the simple method
  
# # selecting optimal value of 'k'
# # using elbow method
  
# # wcss - within cluster sum of
# # squared distances
# wcss = {}

# for j in range(2,elbow_limit+1):
#     model = KMeans(n_clusters=j)
#     model.fit(df_locations)
#     wcss[j] = model.inertia_

# # plotting the wcss values
# # to find out the elbow value
# # plt.plot(wcss.keys(), wcss.values(), 'gs-')
# # plt.xlabel('Values of "k"')
# # plt.ylabel('WCSS')
# # plt.show()

# # SILHOUETTE SCORE METHOD
# # determining the maximum number of clusters 
# # using the simple method
# silhouette_limit = int((df_locations.shape[0]//2)**0.5)
  
# # determing number of clusters
# # using silhouette score method
# df_scores = pd.DataFrame(columns=['Clusters','Score'])
# for k in range(2, silhouette_limit+1):
#     model = KMeans(n_clusters=k)
#     model.fit(df_locations)
#     pred = model.predict(df_locations)
#     score = silhouette_score(df_locations, pred)
#     df_scores.loc[len(df_scores)] = [k, score]

# df_scores.to_excel(r'C:\SeaBikeSafe\seabikesafe\silhouette_scores.xlsx', index = False)

# # clustering the data using Kmeans
# # using k = 40
# kmeans = KMeans(n_clusters=40).fit(df_locations)
# centroids = kmeans.cluster_centers_
# print(centroids)

# plt.scatter(df_locations['X'], df_locations['Y'], c=kmeans.labels_.astype(float), s=200, alpha=0.5)
# plt.scatter(centroids[:,0], centroids[:,1], c='red', s=50)
# plt.show()

# Calculate epsilon
# neigh = NearestNeighbors(n_neighbors=2)
# nbrs = neigh.fit(coords)
# distances, indices = nbrs.kneighbors(coords)
# distances = np.sort(distances, axis=0)
# distances = distances[:,1]
# plt.plot(distances)
# plt.show()

#Computer DBSCAN
# db = DBSCAN(eps=0.05, min_samples=10, algorithm='ball_tree').fit(coords)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

#Number of clusters in labels, ignoring noise if present
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

#Plot
#Black removed and is used for noise instead.
unique_labels = set(labels)
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = labels == k

    xy = coords[class_member_mask & core_samples_mask]
    plt.plot(
        xy[:, 0],
        xy[:, 1],
        "o",
        markerfacecolor=tuple(col),
        markeredgecolor="k",
        markersize=14,
    )

    # xy = coords[class_member_mask & ~core_samples_mask]
    # plt.plot(
    #     xy[:, 0],
    #     xy[:, 1],
    #     "o",
    #     markerfacecolor=tuple(col),
    #     markeredgecolor="k",
    #     markersize=6,
    # )

plt.title("Estimated number of clusters: %d" % n_clusters_)
plt.show()

