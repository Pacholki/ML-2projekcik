import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

class Clusterator():

    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self.read()
        self.geo = ["latitude", "longitude"]

    def read(self):
        if self.file_name[-4:] == ".csv":
            return pd.read_csv(self.file_name)
        else:
            raise ValueError("Unsuported file format")

    def clusterize(self, columns, model=None, ax=None):
        if isinstance(columns, list):
            columns_iter = columns
        else:
            columns_iter = [columns]
        self.check_columns(columns_iter)

        results_df, working_df = self.prep_df(columns_iter)
        model.fit(working_df)
        results_df["cluster"] = model.fit_predict(working_df)
        return self.plot(df=results_df, color_column="cluster", ax=ax)

    def prep_df(self, columns):

        results_df = self.df.copy()[self.geo]
        working_df = self.df.copy()[columns]

        for column in columns:
            mask = self.get_mask(column, working_df)
            results_df = results_df[mask]
            working_df = working_df[mask]

        return results_df, working_df

    def get_mask(self, column, df):
        mask = np.array([True] * len(df))
        return mask

    def plot(self, df, color_column, ax):
        return sns.scatterplot(data=df, x="longitude", y="latitude", hue=df[color_column], legend=False, palette="deep", ax=ax)

    def check_columns(self, columns):
        for column in columns:
            if column not in self.df.columns.to_list():
                raise ValueError(f"\"{column}\" is not a column name. Provide existing column names")

    def normalize(self, df):
        columns = df.columns.to_list()
        for column in columns:
            pass

if __name__ == "__main__":
    file_name = "data/dataset.csv"
    data = Data(file_name=file_name)
    model = KMeans(2)
    data.clusterize(columns=["longitude", "latitude"], model=model)