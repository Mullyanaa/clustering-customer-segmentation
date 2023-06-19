import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#import dataset
df = pd.read_csv('Test.csv')
#hapus kolom yang tidak dipakai
df.drop('ID', axis=1, inplace=True)

#Hapus Null Value
df = df.dropna()

#Merubah Object menjadi Numeric
encode = OrdinalEncoder(dtype=object)
df_encode = pd.DataFrame(encode.fit_transform(df), columns=df.columns)

#standarisasi data
stand = StandardScaler()
stand = stand.fit_transform(df_encode)

#mendeklarasikan Nilai X
X = df_encode

#Header Interface
st.header("isi dataset")
st.write(df)

#Proses Clustering
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

#Menampilkan Elbow
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Mencari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

st.set_option('deprecation.showPyplotGlobalUse', False)
elbo_plot = st.pyplot()

#Membuat slider jumlah K
st.sidebar.subheader("Nilai jumlah K")
clus = st.sidebar.slider("Pilih jumlah cluster: ", 2, 10, 1, 1)

#Membuat pilihan untuk menampilkan grafik
selected_columns = st.sidebar.multiselect('Pilih kolom 1:', df.columns, key='multiselect1') #scatterplot
selected_columns2 = st.sidebar.multiselect('Pilih kolom 2:', df.columns, key='multiselect2') #barplot

x = 'Labels'
y_column = selected_columns2[0] if selected_columns2 else None

#Membuat scatter plot
def k_means(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    if len(selected_columns) >= 2:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=X, x=selected_columns[0], y=selected_columns[1], hue='Labels',
                        markers=True, size='Labels', palette=sns.color_palette('hls', n_clust))

        for label in X['Labels'].unique():
            plt.annotate(label,
                         (X[X['Labels'] == label][selected_columns[0]].mean(),
                          X[X['Labels'] == label][selected_columns[1]].mean()),
                         horizontalalignment='center',
                         verticalalignment='center',
                         size=20, weight='bold',
                         color='black')

        st.header('Cluster Plot')
        st.pyplot()
    else:
        st.write("Pilih 2 kolom untuk menampilkan scatterplot hasil clustering (pilihan pertama = x dan pilihan ke dua = y)")

#membuat barplot
def k_means2(n_clust):
    kmean = KMeans(n_clusters=n_clust).fit(X)
    X['Labels'] = kmean.labels_

    if y_column:
        plt.figure(figsize=(10, 8))
        sns.barplot(data=X, x=x, y=y_column, hue='Labels',
                    palette=sns.color_palette('hls', n_clust))

        for label in X['Labels'].unique():
            plt.annotate(label,
                         (X[X['Labels'] == label][x].mean(),
                          X[X['Labels'] == label][y_column].mean()),
                         ha='center', va='center', size=20, weight='bold', color='black')

        st.header('Cluster Plot')
        st.pyplot()
    else:
        st.write("Pilih kolom 2 untuk menampilkan bar plot hasil clustering")

#menampilkan hasil clustering
    st.header("Hasil Clustering")
    st.write(X)

k_means(clus)
k_means2(clus)
