import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import davies_bouldin_score, silhouette_score

class Preparator():

    def __init__(self, file_name):
        self.file_name = file_name
        self.df = self.read()
        self.GEO = ["latitude", "longitude"]
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

        self.change_to_numeric(["rating", "baths"])
        self.add_popularity()
        self.encode_room_type()

    def change_to_numeric(self, columns):
        for column in columns:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")
            self.df[column] = self.df[column].fillna(4)
    
    def add_popularity(self):
        self.df["popularity"] = self.df["number_of_reviews_ltm"] / self.df["reviews_per_month"]
    
    def encode_room_type(self):
        room_type_map = {
            "Shared room": 0,
            "Private room": 1,
            "Hotel room": 2,
            "Entire home/apt": 3
        }
        self.df["room_type"] = self.df["room_type"].map(room_type_map)

if __name__ == "__main__":
    prep = Preparator("data/dataset.csv")
    print(prep.get_columns())
