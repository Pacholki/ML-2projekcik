import customtkinter as ctk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
from sklearn import mixture
import json

import clustering

class App(ctk.CTk):

    def __init__(self, master=None):
        super().__init__(master)

        self.title("Model Viewer")
        self.geometry("800x600")

        self.data_file_path = "data/dataset.csv"
        with open("data/model_list.json", "r") as file:
            self.model_list = json.load(file)

        self.clusterator = clustering.Clusterator(self.data_file_path)
        self.model=None

        self.model_choice_frame = ModelChoiceFrame(self)
        self.model_choice_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
    def run(self):
        self.mainloop()


class ModelChoiceFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.model_names = list(master.model_list.keys())
        print(self.model_names)
        self.create_widgets()


    def create_widgets(self):
        self.model_choice = ctk.CTkComboBox(master=self, values=self.model_names, command=self.on_model_choice)
        self.model_choice.pack()

    def on_model_choice(self, event):
        print(list(self.master.model_list[event]["params"]))


class StatsFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        pass


class SaveFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        pass


class GraphFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        pass

if __name__ == "__main__":
    app = App()
    app.run()
