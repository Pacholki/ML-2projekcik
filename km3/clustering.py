import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score

class Clusterator():

    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self.read()
        self.geo = ["latitude", "longitude"]
        self.preprocess()

    def read(self):
        if self.file_name[-4:] == ".csv":
            return pd.read_csv(self.file_name)
        else:
            raise ValueError("Unsuported file format")

    def get_columns(self):
        return self.df.columns.to_list()

    def preprocess(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")
        self.df["rating"] = self.df["rating"].fillna(4)
        self.df["baths"] = pd.to_numeric(self.df["baths"], errors="coerce")
        self.df["baths"] = self.df["baths"].fillna(0)
        self.df["popularity"] = self.df["number_of_reviews_ltm"] / self.df["reviews_per_month"]
        self.encode_room_type()
    
    def encode_room_type(self):
        room_type_map = {
            "Shared room": 0,
            "Private room": 1,
            "Hotel room": 2,
            "Entire home/apt": 3
        }
        self.df["room_type"] = self.df["room_type"].map(room_type_map)
        

    def clusterize(self, columns, model=None):
        if isinstance(columns, list):
            columns_iter = columns
        else:
            columns_iter = [columns]
        self.check_columns(columns_iter)

        results_df, working_df = self.prep_df(columns_iter)
        results_df["cluster"] = model.fit_predict(working_df)

        self.results_df = results_df
        self.working_df = pd.DataFrame(working_df)

    def prep_df(self, columns):

        results_df = self.df.copy()
        working_df = self.df.copy()[columns]
        
        for column in columns:
            mask = self.get_mask(column, working_df)
            results_df = results_df[mask]
            working_df = working_df[mask]
            self.repair(column, working_df)

        working_df = preprocessing.scale(working_df)
        return results_df, working_df

    def get_mask(self, column, df):
        if column == "price":
            return np.array(df["price"] < 10000)
        if column == "beds":
            return np.array(df["beds"] < 18)
            # my nie klasteryzujemy domów publicznych
        return np.array([True] * len(df))

    def repair(self, column, df):
        if column == "minimum_nights":
            df.loc[df["minimum_nights"] > 365, "minimum_nights"] = 365
        if column in ["price", "number_of_reviews", "reviews_per_month", "calculated_host_listings_count", "popularity"]:
            df[column] = np.log1p(df[column])

    def plot(self, df=None, pca=False, color_column="cluster", palette="viridis", ax=None, show_values=None):
        if df is None:
            if pca:
                df = self.pca_result_df
            else:
                df = self.results_df

        fig, ax = plt.subplots()
        legend_setting = "brief" if show_values is not None else False
        plot = sns.scatterplot(data=df, x="longitude", y="latitude", hue=df[color_column], legend=legend_setting, palette=palette, ax=ax, alpha=0.7, sizes=(0.5, 0.5))
        plt.show()
        if not show_values:
            return plot
        
        if isinstance(show_values, list):
            show_values_iter = show_values
        else:
            show_values_iter = [show_values]

        for column in show_values_iter:
            self.print_desc_table(df=df, column=column, group_column=color_column)
        
        return plot
    
    def plot_explainded_variance(self, columns):
        scaler = StandardScaler()
        data = self.df
        data = scaler.fit_transform(data[columns])
        data = pd.DataFrame(data, columns=columns)
        pca = PCA()
        pca.fit(data)
        per_var = np.round(pca.explained_variance_ratio_*100, decimals = 1)

        plt.figure(figsize = (10,6))
        plt.plot(range(1, len(per_var)+1), per_var.cumsum(), marker = "o", linestyle = "--")
        plt.grid()
        plt.ylabel("Percentage Cumulative of Explained Variance")
        plt.xlabel("Number of Components")
        plt.title("Explained Variance by Component")
        plt.show()
    
    def pca_plot(self, model, columns, df=None, n_components=2, ax=None):
        if not df:
            df = self.working_df
        
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(df)
        pca_2 = PCA(n_components=2)
        pca_result_2 = pca_2.fit_transform(df)
        pca_df = pd.DataFrame(pca_result_2, columns=["PC1", "PC2"])
        pca_df["cluster"] = model.fit_predict(pca_result)
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(pca_df['PC1'], pca_df['PC2'], c=pca_df['cluster'], cmap='viridis', alpha=0.7)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(*scatter.legend_elements(), title='Clusters')
        plt.title(f'{model.__class__.__name__} Clustering Visualization in 2D')
        plt.show()

                
        self.pca_result_df = self.results_df.copy()
        self.pca_result_df['cluster'] = model.fit_predict(pca_result)
    
    def plot_cluster_distribution(self, df=None, column="cluster", ax=None, pca=False):
        if not df:
            if pca:
                df = self.pca_result_df
            else:
                df = self.results_df

        sns.countplot(x=column, data=df, hue=column, ax=ax, palette='viridis')
        plt.title('Cluster Distribution')
        plt.show()

    def plot_metric_scores(self, n_components, columns, max_clusters=12, df=None):
        if not df:
            df = self.results_df

        pca = PCA(n_components=n_components)
        df = df[columns]
        df = StandardScaler().fit_transform(df)
        pca_result = pca.fit_transform(df)
        wcss = self.wcss_score_counter(pca_result, max_clusters)
        silhouette = self.silhouette_score_counter(pca_result, max_clusters)
        davies_bouldin_score = self.davies_bouldin_score_counter(pca_result, max_clusters)
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        
        ax[0].plot(range(1, max_clusters), wcss)
        ax[0].set_title("WCSS")
        ax[0].set_xlabel("Number of clusters")
        ax[0].set_ylabel("WCSS")
        
        ax[1].plot(range(2, max_clusters), silhouette)
        ax[1].set_title("Silhouette Score")
        ax[1].set_xlabel("Number of clusters")
        ax[1].set_ylabel("Silhouette Score")

        ax[2].plot(range(2, max_clusters), davies_bouldin_score)
        ax[2].set_title("Davies Bouldin Score")
        ax[2].set_xlabel("Number of clusters")
        ax[2].set_ylabel("Davies Bouldin Score")
        
        plt.tight_layout()
        plt.show()

    def wcss_score_counter(self, df, max_clusters):
        wcss = []
        for i in range(1, max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=311)
            kmeans.fit(df)
            wcss.append(kmeans.score(df)*-1)
        return wcss
    
    def silhouette_score_counter(self, df, max_clusters):
        silhouette = []
        for i in range(2, max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=311)
            kmeans.fit(df)
            silhouette.append(silhouette_score(df, kmeans.labels_))
        return silhouette
    
    def davies_bouldin_score_counter(self, df, max_clusters):
        davies_bouldin = []
        for i in range(2, max_clusters):
            kmeans = KMeans(n_clusters=i, random_state=311)
            kmeans.fit(df)
            davies_bouldin.append(davies_bouldin_score(df, kmeans.labels_))
        return davies_bouldin
    
    def plot_features_distributiony_by_clusters(self, columns, pca=False, df=None, cluster_column="cluster", ax=None):
        if not df:
            if pca:
                df = self.pca_result_df
            else:
                df = self.results_df
        self.check_columns(columns)
        ## i want grid layout (2 plots each row)
        fig, ax = plt.subplots(len(columns)//2 + 1, 2, figsize=(15, 15))
        ax = ax.flatten()
        for i, column in enumerate(columns):
            sns.boxplot(data=df, hue=cluster_column, y=column, ax=ax[i], palette='viridis')            
            ax[i].set_title(f'{column} by cluster')
        plt.tight_layout()
        plt.show()


    def print_desc_table(self, df, column, group_column):
        print(f"Values of column {column}:")
        print(df.groupby(group_column)[column].agg(["min", "max", "mean"]).reset_index())

    def check_columns(self, columns):
        for column in columns:
            if column not in self.df.columns.to_list():
                raise ValueError(f"\"{column}\" is not a column name. Provide existing column names")

    def normalize(self, df):
        columns = df.columns.to_list()
        for column in columns:
            pass

    def boxplotter(self, column, df=None, ax=None):
        if df is None:
            df = self.working_df
        return sns.boxplot(data=df, x=column, ax=ax)


if __name__ == "__main__":
    file_name = "data/dataset.csv"
    clusterator = Clusterator(file_name=file_name)
    model = KMeans(2)
    # data.clusterize(columns=["longitude", "latitude"], model=model)
    clusterator.clusterize(columns=["price", "latitude", "longitude"], model=model)
