import customtkinter as ctk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
from sklearn import mixture
import json
import os

import clustering

class App(ctk.CTk):

    def __init__(self, master=None):
        super().__init__(master)

        self.title("Model Viewer")
        self.geometry("1200x800")

        self.data_file_path = "data/dataset.csv"
        self.clusterator = clustering.Clusterator(self.data_file_path)
        self.columns = self.clusterator.get_columns()
        print(self.columns)
        self.models = self.read_models()
        self.model=None

        self.grid_rowconfigure(0, weight=2)
        self.grid_rowconfigure(1, weight=4)
        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.model_choice_frame = ModelChoiceFrame(self)
        self.model_choice_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        self.stats_frame = StatsFrame(self)
        self.stats_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        self.graph_frame = GraphFrame(self)
        self.graph_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")
        self.save_frame = SaveFrame(self)
        self.save_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
    def run(self):
        self.mainloop()

    def read_models(self):
        try:
            with open("data/default_models.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            with open("data/models.json", "r") as file:
                return json.load(file)
    
    def models_revert_to_default(self):
        try:
            os.remove("data/models.json")
            print(f"Modesl reverted to default")
        except FileNotFoundError:
            print("No custom models found")
        

class ModelChoiceFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.model_names = list(master.models.keys())
        self.param_frame = ctk.CTkFrame(master=self)
        self.create_widgets()
        self.param_frame.pack(pady=20, padx=20, fill="both", expand=True)


    def create_widgets(self):
        self.model_choice = ctk.CTkComboBox(master=self, values=self.model_names, command=self.on_model_choice)
        self.model_choice.pack()

    def on_model_choice(self, model_name):
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        params = {k: v for k, v in self.master.models[model_name]["params"].items() if v["active"]}
        for param, details in params.items():
            label = ctk.CTkLabel(master=self.param_frame, text=param)
            field = ctk.CTkEntry(master=self.param_frame, placeholder_text=details["value"])
            label.pack()
            field.pack()

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
