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
        self.columns = self.read_columns()
        self.models = self.read_models()

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

    def read_columns(self):
        try:
            with open("data/columns.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError("No columns file found")

    def read_models(self):
        try:
            with open("data/models.json", "r") as file:
                return json.load(file)
        except FileNotFoundError:
            try:
                with open("data/backup_models.json", "r") as file:
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
        self.model_names = list(self.master.models.keys())
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

        self.generate_column_checkboxes()

        self.on_model_choice(self.model_choice.get())

    def generate_column_checkboxes(self):

        self.column_checkboxes = {}
        self.value_checkboxes= {}

        for i, column in enumerate(self.master.columns):
            checkbutton = ctk.CTkCheckBox(master=self.column_frame, text=column)
            checkbutton.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            self.column_checkboxes[column] = checkbutton

        for i, column in enumerate(self.master.columns):
            checkbutton = ctk.CTkCheckBox(master=self.value_frame, text=column)
            checkbutton.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            self.value_checkboxes[column] = checkbutton

    def on_model_choice(self, model_name):
        self.generate_param_entries(model_name)

    def calculate(self):
        model_name = self.model_choice.get()
        self.generate_param_entries(self.model_choice.get())
        params = self.get_active_params(model_name)
        args = ""
        for param, details in params.items():
            print(details["value"])
            args += f"{param}={details['value']},"
        args = args[:-1]

        model = globals()[self.model_choice.get()](args)
        print(model)

    def generate_param_entries(self, model_name):
        for widget in self.param_frame.winfo_children():
            widget.destroy()

        params = self.get_active_params(model_name)
        self.entries = {}
        for i, (param, details) in enumerate(params.items()):
            label = ctk.CTkLabel(master=self.param_frame, text=param)
            entry = ctk.CTkEntry(master=self.param_frame, placeholder_text=details["value"])
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")
            entry.grid(row=i, column=1, padx=5, pady=5, sticky="ew")
            entry.bind("<KeyRelease>", lambda e, p=param: self.save_entry(e, p))
            self.entries[param] = entry

    def get_active_params(self, model_name):
        return {k: v for k, v in self.master.models[model_name]["params"].items() if v["active"]}

    def save_entry(self, e, param):
        entry = self.entries[param]
        value = entry.get()
        if value:
            self.master.models[self.model_choice.get()]["params"][param]["value"] = value
            print(value)
        else:
            self.master.models[self.model_choice.get()]["params"][param]["value"] = self.master.models[self.model_choice.get()]["params"][param]["default"]


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
