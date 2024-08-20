import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import numpy as np
import folium
from folium.plugins import HeatMap


file_path = 'covid_19_clean_complete.csv'
data = pd.read_csv(file_path)


coordinates = data[['Lat', 'Long']].values


dbscan = DBSCAN(eps=10.0, min_samples=50)
clusters = dbscan.fit_predict(coordinates)


data['Cluster'] = clusters


plt.figure(figsize=(12, 8))
unique_clusters = np.unique(clusters)


colors = plt.cm.get_cmap('tab20', len(unique_clusters))

for cluster in unique_clusters:
    cluster_points = data[data['Cluster'] == cluster]
    plt.scatter(cluster_points['Long'], cluster_points['Lat'], label=f'Cluster {cluster}', s=10, c=[colors(cluster)])

plt.title('DBSCAN Clusters of COVID-19 Cases')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.grid(True)
plt.show()


mapa = folium.Map(location=[0, 0], zoom_start=2)


heatmap_confirmed = data[['Lat', 'Long', 'Confirmed']].dropna()
heatmap_deaths = data[['Lat', 'Long', 'Deaths']].dropna()
heatmap_recovered = data[['Lat', 'Long', 'Recovered']].dropna()


HeatMap(data=heatmap_confirmed, radius=15, max_zoom=13, name='Confirmed Cases').add_to(mapa)


HeatMap(data=heatmap_deaths, radius=15, max_zoom=13, name='Deaths').add_to(mapa)


HeatMap(data=heatmap_recovered, radius=15, max_zoom=13, name='Recovered').add_to(mapa)


folium.LayerControl().add_to(mapa)


mapa.save('covid19_heatmap_layers.html')