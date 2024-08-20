import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


file_path = 'covid_19_clean_complete.csv'
data = pd.read_csv(file_path)


clustering_data = data[['Confirmed', 'Deaths', 'Recovered']].apply(pd.to_numeric, errors='coerce')
clustering_data = clustering_data.dropna()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(clustering_data)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(scaled_data)


clustering_data['Cluster'] = kmeans.labels_


plt.figure(figsize=(18, 6))


plt.subplot(1, 3, 1)
plt.scatter(clustering_data['Confirmed'], clustering_data['Deaths'], c=clustering_data['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Confirmed')
plt.ylabel('Deaths')
plt.title('Confirmed vs Deaths')


plt.subplot(1, 3, 2)
plt.scatter(clustering_data['Confirmed'], clustering_data['Recovered'], c=clustering_data['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Confirmed')
plt.ylabel('Recovered')
plt.title('Confirmed vs Recovered')


plt.subplot(1, 3, 3)
plt.scatter(clustering_data['Deaths'], clustering_data['Recovered'], c=clustering_data['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Deaths')
plt.ylabel('Recovered')
plt.title('Deaths vs Recovered')

plt.suptitle('Pairwise Relationships between Confirmed, Deaths, and Recovered')
plt.tight_layout()
plt.show()