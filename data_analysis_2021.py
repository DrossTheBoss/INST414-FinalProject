import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

#Loads the datasets
adp_data = pd.read_csv("2021_adp_data.csv")
weekly_data = pd.read_csv("2021_weekly_points_data.csv")

#adds a column that has the positon with the number lable
adp_data['position_prefix'] = adp_data['POS'].str.extract(r'([A-Za-z]+)')

#Currently the source does not have data for the columns MFL, Fantrax and FFC
#So we are going to drop those columns to clean up the data
adp_data = adp_data.drop(columns=['MFL', 'Fantrax', 'FFC'])

#we will be more focused on avg and ttl
weekly_data = weekly_data.drop(columns=['Week 1', 'Week 2', 'Week 3', 'Week 4'])
weekly_data = weekly_data.drop(columns=['Week 5', 'Week 6', 'Week 7', 'Week 8'])
weekly_data = weekly_data.drop(columns=['Week 9', 'Week 10', 'Week 11', 'Week 12'])
weekly_data = weekly_data.drop(columns=['Week 13', 'Week 14', 'Week 15', 'Week 16'])
weekly_data = weekly_data.drop(columns=['Week 17', 'Week 18'])

print("Updated data:")
print(adp_data.head())
print(weekly_data.head())
print("\n")

positions = ["QB", "RB", "WR", "TE", "K", "DST"]
weekly_dfs = {}
adp_dfs = {}

#Filters through every position, making a DataFrame for that position
for position in positions:
   
    weekly_current_position = weekly_data[weekly_data['Pos'] == position].copy()
    weekly_dfs[position] = weekly_current_position

    adp_current_position = adp_data[adp_data["position_prefix"] == position].copy()
    adp_dfs[position] = adp_current_position

for key in weekly_dfs:
    print(key + " comparision: \n")
    print(weekly_dfs[key].head())
    print(adp_dfs[key].head())
    print("\n")

#creates tiers for the given position
def tier_position(position_to_tier):
   
    feature_columns = ['AVG', 'TTL']
    data_to_cluster = weekly_dfs[position_to_tier][feature_columns]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data_to_cluster)

    num_clusters = 10
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    weekly_dfs[position_to_tier]['cluster'] = kmeans.fit_predict(scaled_data)

    cluster_avg = weekly_dfs[position_to_tier].groupby('cluster')['AVG'].mean()
    top_clusters = cluster_avg.sort_values(ascending=False).head(5).index

    #Prints out the top 5 tiers and the players in the tier
    for cluster_idx in top_clusters:
        print(f"Cluster {cluster_idx} - Average AVG: {cluster_avg[cluster_idx]:.2f}")
        cluster_players = weekly_dfs[position_to_tier][weekly_dfs[position_to_tier]['cluster'] == cluster_idx]
        print(cluster_players[['Player', 'AVG']].sort_values(by='AVG', ascending=False))
        print("\n")

for position in positions:
    tier_position(position)