import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


file_path = 'covid_19_clean_complete.csv'
data = pd.read_csv(file_path)


countries_coords = data.groupby('Country/Region')[['Lat', 'Long']].mean().reset_index()


china_coords = countries_coords[countries_coords['Country/Region'] == 'China'][['Lat', 'Long']]


k = 10


knn = NearestNeighbors(n_neighbors=k+1)
knn.fit(countries_coords[['Lat', 'Long']])
distances, indices = knn.kneighbors(china_coords)


nearest_neighbors = countries_coords.iloc[indices[0][1:]]['Country/Region'].values


china_data = data[data['Country/Region'] == 'China']
neighbors_data = data[data['Country/Region'].isin(nearest_neighbors)]


china_data = china_data.groupby('Date').sum().reset_index()
correlations = pd.DataFrame(columns=['Country', 'Confirmed', 'Deaths', 'Recovered'])


for country in nearest_neighbors:
    neighbor_data = neighbors_data[neighbors_data['Country/Region'] == country].groupby('Date').sum().reset_index()
    
    if len(neighbor_data) == 0:
        continue

    merged_data = pd.merge(china_data, neighbor_data, on='Date', suffixes=('_China', f'_{country}'))

    confirmed_corr = merged_data['Confirmed_China'].corr(merged_data[f'Confirmed_{country}'])
    deaths_corr = merged_data['Deaths_China'].corr(merged_data[f'Deaths_{country}'])
    recovered_corr = merged_data['Recovered_China'].corr(merged_data[f'Recovered_{country}'])
    
    
    if pd.isna(confirmed_corr):
        confirmed_corr = 0
    if pd.isna(deaths_corr):
        deaths_corr = 0
    if pd.isna(recovered_corr):
        recovered_corr = 0
    
    correlations = correlations._append({
        'Country': country,
        'Confirmed': confirmed_corr,
        'Deaths': deaths_corr,
        'Recovered': recovered_corr
    }, ignore_index=True)


print(correlations)


labels = correlations['Country']
confirmed_corrs = correlations['Confirmed']
deaths_corrs = correlations['Deaths']
recovered_corrs = correlations['Recovered']

fig, ax = plt.subplots(1, 3, figsize=(18, 6))

ax[0].pie(confirmed_corrs, labels=labels, autopct='%1.1f%%')
ax[0].set_title('Correlacion de Casos Confirmados con China')

ax[1].pie(deaths_corrs, labels=labels, autopct='%1.1f%%')
ax[1].set_title('Correlacion de Muertes con China')

ax[2].pie(recovered_corrs, labels=labels, autopct='%1.1f%%')
ax[2].set_title('Correlacion de Casos Recuperados con China')

plt.show()
