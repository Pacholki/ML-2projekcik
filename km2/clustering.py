import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import preprocessing

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

    def preprocess(self):
        self.df = self.df.dropna()
        self.df = self.df.drop_duplicates()
        self.df["rating"] = pd.to_numeric(self.df["rating"], errors="coerce")
        self.df["rating"] = self.df["rating"].fillna(0)
        self.df["baths"] = pd.to_numeric(self.df["baths"], errors="coerce")
        self.df["baths"] = self.df["baths"].fillna(0)
        self.df["popularity"] = self.df["number_of_reviews_ltm"] / self.df["reviews_per_month"]

    def clusterize(self, columns, model=None):
        if isinstance(columns, list):
            columns_iter = columns
        else:
            columns_iter = [columns]
        self.check_columns(columns_iter)

        results_df, working_df = self.prep_df(columns_iter)
        results_df["cluster"] = model.fit_predict(working_df)

        self.results_df = results_df
        self.working_df = pd.DataFrame(working_df, columns=columns_iter)

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

    def plot(self, df=None, color_column="cluster", palette="deep", ax=None, show_values=None):
        if not df:
            df = self.results_df

        legend_setting = "brief" if show_values is not None else False
        plot = sns.scatterplot(data=df, x="longitude", y="latitude", hue=df[color_column], legend=legend_setting, palette=palette, ax=ax, alpha=0.7)

        if not show_values:
            return plot
        
        if isinstance(show_values, list):
            show_values_iter = show_values
        else:
            show_values_iter = [show_values]

        for column in show_values_iter:
            self.print_desc_table(df=self.results_df, column=column, group_column=color_column)
        
        return plot

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