import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances,manhattan_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import time
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


def clustering_md(train_arr, centroids):    
    ed_matrix = manhattan_distances(train_arr, centroids)
    nearest_centroid = []
    for i in range(ed_matrix.shape[0]):
        c = np.argmin(ed_matrix[i])
        nearest_centroid.append(c)
    return nearest_centroid



def clustering_eu(train_arr, centroids):    
    ed_matrix = euclidean_distances(train_arr, centroids)
    nearest_centroid = []
    for i in range(ed_matrix.shape[0]):
        c = np.argmin(ed_matrix[i])
        nearest_centroid.append(c)
    return nearest_centroid


def clustering_cos(train_arr, centroids):    
    ed_matrix = cosine_similarity(train_arr, centroids)
    nearest_centroid = []
    for i in range(ed_matrix.shape[0]):
        c = np.argmax(ed_matrix[i])
        nearest_centroid.append(c)
    return nearest_centroid


def calc_centroids(train_arr, nearest_centroid, centroids):
    cluster_d = list()
    all_cluster_d = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    new_centroids = list()
    new_df = pd.concat([pd.DataFrame(train_arr), pd.DataFrame(nearest_centroid, columns=['cluster'])], axis=1)    
    new_df_arr = np.array(new_df['cluster'])
    for c in set(new_df_arr):        
        thiscluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]        
        temp = np.array(centroids[c])
        temp = temp.reshape(1,-1)        
#         cluster_d = manhattan_distances(thiscluster, temp)
        cluster_d = euclidean_distances(thiscluster, temp)
#         cluster_d = cosine_similarity(thiscluster, temp)
        for d in cluster_d:
            all_cluster_d[c] += d * d        
        cluster_mean = thiscluster.mean(axis=0)        
        new_centroids.append(cluster_mean)    
    return new_centroids, all_cluster_d



def runner(k):
    
    df = pd.read_csv("1614633448_8303087_image_new_test.txt", header=None)
    arr = np.array(df)
    arr = arr.astype(float)
    
    #normalization of data using minmax scaler
    scaler = MinMaxScaler()
    scaled_arr = scaler.fit_transform(arr)
#     train_arr = scaled_arr
    
    np.random.seed(42)
    rndperm = np.random.permutation(df.shape[0])
    
    pca_70 = PCA(n_components=70)
    pca_result = pca_70.fit_transform(scaled_arr)
#     train_arr = pca_result
    
    
    print('Cumulative explained variation for 70 principal components: {}'.format(np.sum(pca_70.explained_variance_ratio_)))

    
    tsne = TSNE(n_components = 2, perplexity = 50, init = 'pca', random_state=0)
    train_arr = tsne.fit_transform(pca_result)
    
    c_index = random.sample(range(0, len(train_arr)), k)    
    centroids = []
    
    for i in c_index:
        centroids.append(train_arr[i])
    centroids = np.array(centroids)    
    
    sse = []
    iterations = 25
    nearest_centroid = []
    for i in range(iterations):
#         nearest_centroid = clustering_md(train_arr, centroids)
        nearest_centroid = clustering_eu(train_arr, centroids)
#         nearest_centroid = clustering_cos(train_arr, centroids)
        centroids, all_cluster_d = calc_centroids(train_arr, nearest_centroid, centroids)
        sse.append(sum(all_cluster_d))                
    new_df = pd.concat([pd.DataFrame(train_arr), pd.DataFrame(nearest_centroid, columns=['cluster'])], axis=1)
    new_df.replace({0:1, 1:2,2:3,3:4,4:5,5:6,6:7,7:8,8:9,9:10}, inplace=True)
    new_df.to_csv('test_pca_70_eu.csv',columns=['cluster'], index =False, header = False)
    
    new_df['pca-one'] = pca_result[:,0]
    new_df['pca-two'] = pca_result[:,1] 
    new_df['pca-three'] = pca_result[:,2]
    
    new_df['tsne-2d-one'] = train_arr[:,0]
    new_df['tsne-2d-two'] = train_arr[:,1]
    
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="cluster",
    palette=sns.color_palette("hls", 10),
    data=new_df,
    legend="full",
    alpha=0.3
    )
    
    ax2 = plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="cluster",
        palette=sns.color_palette("hls", 10),
        data=new_df,
        legend="full",
        alpha=0.3
    )
    
    ax = plt.figure(figsize=(16,10)).gca(projection='3d')
    ax.scatter(
        xs=new_df["pca-one"], 
        ys=new_df["pca-two"], 
        zs=new_df["pca-three"], 
        c=new_df["cluster"], 
        cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()


    
    print("minimum sse for k = {} is {}".format(k, np.min(sse)))
    
#     return np.min(sse)
          
    plt.figure()
    plt.plot(range(iterations), sse, 'bx-')
    plt.xlabel('iterations')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal iterations')
    plt.show()
    
    
    
    
    
# start = time.time()
runner(10)
# print("Execution time = ", time.time() - start)

# sse_all = []

# for k in range(2,21,2):
#     sse_all.append(runner(k))
    
# plt.figure()
# plt.plot(range(2,21,2), sse_all, 'bx-')
# plt.xlabel('Values of K')
# plt.ylabel('min SSE for that K')
# plt.title('min SSE vs K')
# plt.show()