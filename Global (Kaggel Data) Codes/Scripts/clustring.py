from configs import * 
from load_explore_data import *
from preprocessing import * 


def elbow_method(df, columns, random_state=RANDOM_STATE):
    inertia = []
    K = range(1, 10)

    for k in K:
        model = KMeans(n_clusters=k, random_state=random_state)
        model.fit(df[columns])
        inertia.append(model.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(K, inertia, 'bo-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia (Within-cluster sum of squares)')
    plt.title('Elbow Method for Optimal k')
    plt.show()


def clustering(data, columns, k, random_state=RANDOM_STATE):
    kmeans = KMeans(n_clusters=k, random_state=random_state)
    
    # Fit and Predict on Train
    data = data.copy()
    
    data['Cluster'] = kmeans.fit_predict(data[columns])
    
    # Calculate score based on training data
    score = silhouette_score(data[columns], data['Cluster'])
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=columns)
    
    return data, kmeans, score, centroids


def map_clusters(df):
    cluster_names = {
        0: "Low Agricultural Viability Cluster",
        1: "High Agricultural Performance Cluster",
        2: "Balanced Agricultural Productivity Cluster"
    }
    
    # Replace numeric labels with names
    df['Cluster_Name'] = df['Cluster'].map(cluster_names)
    
    return df 
