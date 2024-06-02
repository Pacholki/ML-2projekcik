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
        self.minsize(1200, 800)

        self.data_file_path = "data/dataset.csv"
        self.clusterator = clustering.Clusterator(self.data_file_path)
        self.columns = self.clusterator.get_columns()
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
            with open("data/models.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            try:
                with open("data/default_models.json", "r") as file:
                    return json.load(file)
            except FileNotFoundError:
                raise FileNotFoundError("No models file found")
    
    def models_revert_to_default(self):
        try:
            os.remove("data/models.json")
            print(f"Modesl reverted to default")
        except FileNotFoundError:
            print("No custom models found")
    
    def save_models(self):
        with open("data/models.json", "w") as file:
            json.dump(self.models, file, indent=4)
        

class ModelChoiceFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.model_names = list(master.models.keys())
        self.create_widgets()


    def create_widgets(self):
        self.model_choice = ctk.CTkComboBox(master=self, values=self.model_names, command=self.on_model_choice)
        self.model_choice.configure(width=250, height=30)
        self.model_choice.grid(row=0, column=0, columnspan=3, padx=5, pady=5)

        self.param_frame = ctk.CTkFrame(master=self)
        self.column_frame = ctk.CTkFrame(master=self)
        self.value_frame = ctk.CTkFrame(master=self)
        self.param_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        self.column_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        self.value_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")

        self.calculate_button = ctk.CTkButton(master=self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=2, column=0, columnspan=3, padx=5, pady=5, sticky="nsew")

        self.on_model_choice(self.model_choice.get())

    def on_model_choice(self, model_name):
        print("siema")
        for widget in self.param_frame.winfo_children():
            widget.destroy()
        self.params = {k: v for k, v in self.master.models[model_name]["params"].items() if v["active"]}
        self.entries = {}
        for i, (param, details) in enumerate(self.params.items()):
            label = ctk.CTkLabel(master=self.param_frame, text=param)
            entry = ctk.CTkEntry(master=self.param_frame, placeholder_text=details["value"])
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            self.entries[param] = entry

    def calculate(self):
        for param, entry in self.entries.items():
            value = entry.get()
            if value:
                self.master.models[self.model_choice.get()]["params"][param]["value"] = value


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
    app.save_models()
