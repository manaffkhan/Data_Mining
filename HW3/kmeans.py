import pandas as pd
import numpy as np
import random
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import os


def clustering_eu(train_arr, centroids):
    
    ed_matrix = euclidean_distances(train_arr, centroids)
    nearest_centroid = []
#     mean_sum=[0,0,0]
    for i in range(ed_matrix.shape[0]):
        c = np.argmin(ed_matrix[i])
        nearest_centroid.append(c)
#         mean_sum[c] += ed_matrix[i][c]
#     print("nearest Centroid ", nearest_centroid)
    return nearest_centroid


def clustering_cos(train_arr, centroids):
    
    ed_matrix = cosine_similarity(train_arr, centroids)
    nearest_centroid = []
#     mean_sum=[0,0,0]
    for i in range(ed_matrix.shape[0]):
        c = np.argmax(ed_matrix[i])
        nearest_centroid.append(c)
#         mean_sum[c] += ed_matrix[i][c]
#     print("nearest Centroid ", nearest_centroid)
    return nearest_centroid
    
def calc_centroids(train_arr, nearest_centroid, centroids):
    cluster_d = [] 
    all_cluster_d = [0.0,0.0,0.0]
    new_centroids = []
#     print("nearest Centroid2 ", nearest_centroid)
    new_df = pd.concat([pd.DataFrame(train_arr), pd.DataFrame(nearest_centroid, columns=['cluster'])], axis=1)
    print(new_df)
    new_df_arr = np.array(new_df['cluster'])
    for c in set(new_df_arr):
        print("c  ", c)
        thiscluster = new_df[new_df['cluster'] == c][new_df.columns[:-1]]
        print("centroids.  ",centroids[c])
        temp = np.array(centroids[c])
        temp = temp.reshape(1,-1)
        print("temp  ", temp)
        cluster_d = euclidean_distances(thiscluster, temp)
        for d in cluster_d:
            all_cluster_d[c] += d*d
        print("thiscluster ", thiscluster)
        cluster_mean = thiscluster.mean(axis=0)
        print("Cluster mean ", cluster_mean)
        new_centroids.append(cluster_mean)
    print("new centroids :", new_centroids)
    return new_centroids, all_cluster_d

def runner():
    
    df = pd.read_table("1614633221_7835212_iris_new_data.txt", header=None, skip_blank_lines=False, delim_whitespace=True)
#     print(df)
#     convert the data into a numpy array
    train_arr = np.array(df)
#     print(arr)
    no_clusters = 3
    c_index = random.sample(range(0, len(df)), no_clusters)
    print(c_index)
    centroids = []
    for i in c_index:
        centroids.append(df.loc[i])
    centroids = np.array(centroids)
    print(centroids)
    sse = []
    iterations = 10
    nearest_centroid = []
    for i in range(iterations):
        nearest_centroid = clustering_eu(train_arr, centroids)
#         nearest_centroid = clustering_cos(train_arr, centroids)
        centroids, all_cluster_d = calc_centroids(train_arr, nearest_centroid, centroids)
        sse.append(all_cluster_d[0] + all_cluster_d[1] +all_cluster_d[2])
        
        #print(centroids)
#         plt.figure()
#         plt.scatter(np.array(centroids)[:, 0], np.array(centroids)[:, 1], color='black')
#         plt.scatter(train_arr[:, 0], train_arr[:, 1], alpha=0.1)
#         plt.show()
    new_df = pd.concat([pd.DataFrame(train_arr), pd.DataFrame(nearest_centroid, columns=['cluster'])], axis=1)
    print(new_df)
    new_df_arr = np.array(new_df['cluster'])
 
#     f1 = open("iris_result_cos.txt", "w")
#     for i in new_df_arr:
# #         print(str(i))
#         f1.write(str(i+1))
#         f1.write(os.linesep)
#     f1.close()
    new_df.replace({0:1,1:2,2:3}, inplace = True)
#     new_df['cluster'].to_csv('iris_result_cos.csv', index = False, header = False)
    new_df['cluster'].to_csv('iris_result_eu.csv', index = False, header = False)
    
    plt.figure()
    plt.plot(range(iterations), sse, 'bx-')
    plt.xlabel('iterations')
    plt.ylabel('SSE')
    plt.title('The Elbow Method showing the optimal iterations')
    plt.show()
    
    


    
    
runner()