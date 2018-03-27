# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
from amltlearn.metrics.cluster import davies_bouldin_score
from PyNomaly import loop
np.random.seed(0)
    
num_dict = {}
num_dict['director_name_num'] = {}
num_dict['actor_1_name_num'] = {}
num_dict['actor_2_name_num'] = {}
num_dict['actor_3_name_num'] = {}
num_dict['country_num'] = {}
num_dict['genre_0_num'] = {}
num_dict['genre_1_num'] = {}

# Convert text data into integers
def handle_non_numerical_data(df):
    columns = df.columns.values
    for column in columns:
        text_digit_vals = {}

        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    num_dict["{0}_num".format(column)][unique] = x 
                    x+=1

            df["{0}_num".format(column)] = list(map(convert_to_int, df[column]))

    return df

def import_and_reduce(full):
    #  pandas try to converts object dtype to numeric
    full.convert_objects(convert_numeric=True)    
#    print(full.head())
    
    full['genres'] = full['genres'].str.split('|')
    
    df1 = pd.DataFrame(full.genres.values.tolist()).add_prefix('genre_')
    a = pd.concat([full, df1], axis=1)
    
    
    a['duration'].fillna(a['duration'].mean(), inplace=True)
    a = a.dropna(subset=['genre_1', 'year', 'duration'])
    
    a = a[['director_name','duration','actor_2_name','actor_1_name','actor_3_name','country','year','imdb','genre_0','genre_1']]
    
    a = handle_non_numerical_data(a)
    
    a = a[['director_name_num','duration','actor_2_name_num','actor_1_name_num','actor_3_name_num','country_num','year','imdb','genre_0_num','genre_1_num']]
    
    
    #a.to_csv("num_dataset.csv", index=False)
    
    X_train = np.array(a)
#    y = np.array(a['imdb'])
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, shuffle=False)
    
    std_scaler = RobustScaler()
    std_scaler.fit(X_train)
    data = std_scaler.transform(X_train)
    
    n_samples, n_features = data.shape

    # Applying Principal component analysis
    pca = PCA(n_components=3)
    pca.fit(data)
    reduced_data = pca.transform(data)
    return reduced_data, n_samples, n_features

# Import data from JSON...
full = pd.read_json('dataset.json', orient='records')
reduced_data, n_samples, n_features = import_and_reduce(full)
scores = loop.LocalOutlierProbability(reduced_data).fit()
#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(reduced_data[:,0], reduced_data[:,1], reduced_data[:,2],
#c=scores, cmap='seismic', s=10)
#ax.set_xlabel('X')
#ax.set_ylabel('Y')
#ax.set_zlabel('Z')
#plt.savefig('2d_plots/3d_plot_outlier.png', format='png', dpi=700)
#plt.show()
#plt.clf()
#plt.cla()
#plt.close()

#probabilities = [0.95, 0.90, 0.85]
probabilities = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]

def loop_prob(full, h):
    print("reduced data :", reduced_data.shape)
#    print("reduced data cleaned b4:", reduced_data_cleaned.shape)
    index_to_remove = []
    full = full.assign(loop_score=scores)
    for c, i in enumerate(full.loop_score):
    #    print(i)
        if i > h:
            index_to_remove.append(c)
    reduced_data_cleaned = np.delete(reduced_data, index_to_remove, 0)
            
    print("reduced data cleaned aftr:", reduced_data_cleaned.shape)
#    reduced_data = reduced_data_cleaned
    #reduced_data = reduced_data[reduced_data.loop_score < 0.95]
    
    #print(reduced_data.head())
    
    # Import data from JSON...
    #full_without_outliers = pd.read_json('dataset_without_outliers.json', orient='records')
    
    # Enter number of clusters
#    n_digits = int(input("Enter number of clusters: ".format(n_features)))
    n_digits = 10
    #    labels = y_train
    print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_digits, n_samples, n_features))
    
    # #############################   FUZZY C MEANS  ############################################
    
#    alldata = np.vstack((reduced_data_cleaned[:,0], reduced_data_cleaned[:,1], reduced_data_cleaned[:,2]))

    
    # Fuzzy C Means
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(reduced_data_cleaned.T, n_digits, 2, error=0.0001, maxiter=1000, init=None)
    
    np.savetxt('intit_fpm.csv', u0, delimiter=",")
    #################################### VISUALIZATION for FUZZY CLUSTERING ##################################
    
    cluster_membership = np.argmax(u, axis=0)
    
    f_silhouette_score = silhouette_score(reduced_data_cleaned, cluster_membership)
    
    f_davies_bouldin_score = davies_bouldin_score(reduced_data_cleaned, cluster_membership)
    
    #################################### KMEANS CLUSTERING #############################################
    
    
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10, max_iter=1000)
    kmeans.fit(reduced_data_cleaned)
    
    
    #################################### VISUALIZATION for KMEANS CLUSTERING ###########################
    
    cluster_membership = kmeans.labels_
    
    k_silhouette_score = silhouette_score(reduced_data_cleaned, cluster_membership)
    k_davies_bouldin_score = davies_bouldin_score(reduced_data_cleaned, cluster_membership)
    
    
    print(82 * '_')
    print('FCM Score')
    print('%-9s\t\t\t%.2f' % ("Fuzzy C-Means Silhouette score", f_silhouette_score))
    print('%-9s\t%.2f' % ("Fuzzy C-Means Davies-Bouldin score", f_davies_bouldin_score))
    print(82 * '_')
    
    print(82 * '_')
    print('K-Means Score')
    print('%-9s\t\t\t%.2f' % ("K-Means Clustering Silhouette score", k_silhouette_score))
    print('%-9s\t%.2f' % ("K-Means Clustering Davies-Bouldin score", k_davies_bouldin_score))
    print(82 * '_')
    
    return k_silhouette_score, k_davies_bouldin_score, f_silhouette_score, f_davies_bouldin_score

ksilhouette_met = []
kdavies_bouldin_met = []
fsilhouette_met = []
fdavies_bouldin_met = []

for j in probabilities:
    k_silhouette_score, k_davies_bouldin_score, f_silhouette_score, f_davies_bouldin_score = loop_prob(full, j)
    ksilhouette_met.append(k_silhouette_score)
    kdavies_bouldin_met.append(k_davies_bouldin_score)
    fsilhouette_met.append(f_silhouette_score)
    fdavies_bouldin_met.append(f_davies_bouldin_score)
    
fig = plt.figure(figsize=(11,8))
ax2 = fig.add_subplot(111)

ax2.plot(probabilities, ksilhouette_met, label='K-Means Clustering Silhouette score', color='c', marker='o')
ax2.plot(probabilities, kdavies_bouldin_met, label='K-Means Clustering Davies-Bouldin score', color='g', marker='o')

plt.xticks(probabilities)
plt.xlabel('outlier probabilities')
plt.ylabel('Score')

handles, labels = ax2.get_legend_handles_labels()
lgd = ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
ax2.grid('on')

plt.savefig('k_loop.png')


fig = plt.figure(figsize=(11,8))
ax2 = fig.add_subplot(111)

ax2.plot(probabilities, fsilhouette_met, label='K-Means Clustering Silhouette score', color='c', marker='o')
ax2.plot(probabilities, fdavies_bouldin_met, label='K-Means Clustering Davies-Bouldin score', color='g', marker='o')

plt.xticks(probabilities)
plt.xlabel('outlier probabilities')
plt.ylabel('Score')

handles, labels = ax2.get_legend_handles_labels()
lgd = ax2.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.15,1))
ax2.grid('on')

plt.savefig('f_loop.png')
