import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

collisions_file = r'C:\SeaBikeSafe\seabikesafe\Collisions.csv'
df = pd.read_csv(collisions_file, dtype={"X": float, "Y": float, "PEDCYLCOUNT":int, "SEVERITYCODE":"string"}, low_memory=False, index_col=False)
df = df[(df['PEDCYLCOUNT'] != 0) & ((df["SEVERITYCODE"] == "2" ) | (df["SEVERITYCODE"] == "2b") | (df["SEVERITYCODE"] == "3"))]
coords = pd.DataFrame(columns=['Y','X'])
coords = df[['Y', 'X']]
coords = (coords.dropna()).reset_index(drop=True)
plot_coords = coords
print(plot_coords)
coords = coords.values
kms_per_radian = 6371.0088 
epsilon = 0.1/kms_per_radian

db = DBSCAN(eps=epsilon, min_samples=10, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
cluster_labels = db.labels_
num_clusters= len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
clusters = pd.Series([coords[cluster_labels == n] for n in range(num_clusters)])
print('Number of clusters: {}'.format(num_clusters))

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)
centermost_points = clusters.map(get_centermost_point)

lats, lons = zip(*centermost_points)
rep_points = pd.DataFrame({'lon': lons, 'lat':lats})

fig, ax = plt.subplots(figsize=[10,6])
rs_scatter = ax.scatter(rep_points['lon'], rep_points['lat'], c='#99cc99', edgecolor='None', alpha=0.7, s=120)
df_scatter = ax.scatter(plot_coords['X'], plot_coords['Y'], c='k', alpha=0.9, s=1)
ax.set_title('Full Data Set vs DBSCAN Reduced Set of Seattle Bicycle Collision Data')
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend([df_scatter, rs_scatter], ['Full Set','Reduced Set'], loc='upper right')
plt.show()