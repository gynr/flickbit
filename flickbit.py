from time import time
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import skfuzzy as fuzz
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from PyNomaly import loop

np.random.seed(0)

# Import data from JSON...
full = pd.read_json('dataset.json', orient='records')
print(full.shape)

#  pandas try to converts object dtype to numeric
full.convert_objects(convert_numeric=True)

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

def handle_non_numerical_data_input(df):
    df['director_name'].replace(num_dict['director_name_num'], inplace=True)
    df['actor_1_name'].replace(num_dict['actor_1_name_num'], inplace=True)
    df['actor_2_name'].replace(num_dict['actor_2_name_num'], inplace=True)
    df['actor_3_name'].replace(num_dict['actor_3_name_num'], inplace=True)
    df['country'].replace(num_dict['country_num'], inplace=True)
    df['genre_0'].replace(num_dict['genre_0_num'], inplace=True)
    df['genre_1'].replace(num_dict['genre_1_num'], inplace=True)
    return df

def write_dictionary(file_name, dict_num):
    pd.DataFrame.from_dict(data=dict_num, orient='index').to_csv(file_name, header=False)

full['genres'] = full['genres'].str.split('|')

df1 = pd.DataFrame(full.genres.values.tolist()).add_prefix('genre_')
a = pd.concat([full, df1], axis=1)

# Count NAN in genres and remove with large NAN
objectz = ('Genre_0', 'Genre_1', 'Genre_2', 'Genre_3', 'Genre_4', 'Genre_5', 'Genre_6')
y_pos = np.arange(len(objectz))

performance = []
for i in range(7):
    genre_na = a['genre_{0}'.format(i)].isnull().sum()
    performance.append(genre_na)
1
plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objectz)
plt.ylabel('NaN count')
plt.title('NaN in genres')
plt.savefig('genres_nan.png', format='png', dpi=700)
plt.show()

a['duration'].fillna(a['duration'].mean(), inplace=True)
a = a.dropna(subset=['genre_1', 'year', 'duration'])

a = a[['director_name','duration','actor_2_name','actor_1_name','actor_3_name','country','year','imdb','genre_0','genre_1']]

a = handle_non_numerical_data(a)

write_dictionary('num_dict/director.csv', num_dict['director_name_num'])
write_dictionary('num_dict/actor_1.csv', num_dict['actor_1_name_num'])
write_dictionary('num_dict/actor_2.csv', num_dict['actor_2_name_num'])
write_dictionary('num_dict/actor_3.csv', num_dict['actor_3_name_num'])
write_dictionary('num_dict/country.csv', num_dict['country_num'])
write_dictionary('num_dict/genre_0.csv', num_dict['genre_0_num'])
write_dictionary('num_dict/genre_1.csv', num_dict['genre_1_num'])

a = a[['director_name_num','duration','actor_2_name_num','actor_1_name_num','actor_3_name_num','country_num','year','imdb','genre_0_num','genre_1_num']]


#a.to_csv("num_dataset.csv", index=False)

X = np.array(a)
y = np.array(a['imdb'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0, shuffle=False)

std_scaler = RobustScaler()
std_scaler.fit(X_train)
median = std_scaler.center_
scaling = std_scaler.scale_

rscale = pd.DataFrame(median)
#rscale = rscale.T
rscale.to_csv('median.csv', index=False)
rscale = pd.DataFrame(scaling)
#rscale = rscale.T
rscale.to_csv('scaling.csv', index=False)
    
data = std_scaler.transform(X_train)

n_samples, n_features = data.shape

# Enter number of clusters
n_digits = int(input("Enter number of clusters: ".format(n_features)))

labels = y_train

print("n_digits: %d, \t n_samples %d, \t n_features %d" % (n_digits, n_samples, n_features))

# Applying Principal component analysis
#pca = PCA(n_components=3)
#pca.fit(data)
#reduced_data = pca.transform(data)

# #############################   FUZZY C MEANS  ############################################

colors = ['b', 'orange', 'g', 'r', 'c', 'm', 'y', 'k', 'Brown', 'ForestGreen']

# initial time stamp : Fuzzy
t0 = time()

# Fuzzy C Means
kmeans = KMeans(init='random', n_clusters=n_digits, n_init=10, max_iter=1000)
kmeans.fit(data)
cntr = kmeans.cluster_centers_
np.savetxt('centers.csv', cntr, delimiter=",")

labels = kmeans.labels_
np.savetxt('labels.csv', labels, delimiter=",")

# final time stamp : Fuzzy
t1 = time() - t0

#################################### VISUALIZATION for FUZZY CLUSTERING ##################################

cluster_membership = kmeans.labels_

full = full.assign(center=cluster_membership)
for i in range(n_digits):
    d = full[full.center == i]
    d.to_json('clusters/movie_{0}.json'.format(i), orient='records')
