import customtkinter as ctk
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids
from hdbscan import HDBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import json
import os
from PIL import Image, ImageTk
import io

import clustering

class App(ctk.CTk):

    def __init__(self, master=None):
        super().__init__(master)

        self.title("Model Viewer")
        self.geometry("1200x800")
        self.minsize(1200, 800)
        self.vertical_resolution = 1300
        self.horizontal_resolution = 3400

        self.data_file_path = "data/dataset.csv"
        self.clusterator = clustering.Clusterator(self.data_file_path)
        self.columns = self.read_columns()
        self.models = self.read_models()

        self.model_choice_frame = ModelChoiceFrame(self)
        self.model_choice_frame.grid(row=0, column=0, padx=20, pady=5, sticky="ns")
        self.graph_frame = GraphFrame(self)
        self.graph_frame.grid(row=1, column=0, padx=20, pady=5, sticky="nsew")
        self.stats_frame = StatsFrame(self)
        self.stats_frame.grid(row=0, column=1, rowspan=2, padx=20, pady=5, sticky="nsew")
        # self.save_frame = SaveFrame(self)
        # self.save_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=20, sticky="nsew")
        
        
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

        self.grid_columnconfigure(1, weight=2)
        self.grid_columnconfigure(2, weight=1)
        self.param_frame = ctk.CTkFrame(master=self)
        self.column_frame = ctk.CTkFrame(master=self)
        self.value_frame = ctk.CTkFrame(master=self)
        self.param_frame.grid(row=1, column=0, rowspan=5, padx=5, pady=5, sticky="nsew")
        self.column_frame.grid(row=1, column=1, rowspan=5, padx=5, pady=5, sticky="nsew")
        self.value_frame.grid(row=1, column=2, rowspan=5, padx=5, pady=5, sticky="nsew")

        self.calculate_button = ctk.CTkButton(master=self, text="Calculate", command=self.calculate)
        self.calculate_button.grid(row=0, column=4, padx=5, pady=5, sticky="nsew")

        self.plot_button = ctk.CTkButton(master=self, text="Plot", command=self.plot)
        self.plot_button.grid(row=1, column=4, padx=5, pady=5, sticky="nsew")
        
        self.plot_pca_button = ctk.CTkButton(master=self, text="Plot PCA", command=self.plot_pca)
        self.plot_pca_button.grid(row=2, column=4, padx=5, pady=5, sticky="nsew")

        self.plot_stats_button = ctk.CTkButton(master=self, text="Plot Stats", command=self.plot_stats)
        self.plot_stats_button.grid(row=3, column=4, padx=5, pady=5, sticky="nsew")

        self.n_components_entry = ctk.CTkEntry(master=self, placeholder_text="Number of components")
        self.n_components_entry.grid(row=4, column=4, padx=5, pady=5, sticky="nsew")

        self.status_label = ctk.CTkLabel(master=self, text="")
        self.status_label.grid(row=5, column=4, padx=5, pady=5, sticky="nsew")

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

        self.status_label.configure(text="Calculating...")

        model_name = self.model_choice.get()
        params = self.get_active_params(model_name)
        args = {}
        for param, details in params.items():

            value = details["value"]
            param_type = details.get("type", "str")  # Default to string if type is not provided
            
            if param_type == "int":
                args[param] = int(value)
            elif param_type == "float":
                args[param] = float(value)
            elif param_type == "str":
                args[param] = str(value)
            elif param_type == "NoneType":
                args[param] = None
            else:
                raise ValueError(f"Unsupported parameter type: {param_type}")

        model_class = globals().get(model_name)
        if model_class:
            try:
                model = model_class(**args)
            except TypeError as e:
                print(f"Error instantiating model: {e}")
        else:
            print(f"Model '{model_name}' not found")

        columns = [column for column, checkbox in self.column_checkboxes.items() if checkbox.get()]
        self.master.clusterator.clusterize(model=model, columns=columns)
        self.master.columns = columns
        self.master.model = model

        self.status_label.configure(text="Done")

    def plot(self):
        self.master.graph_frame.plot()
        self.plot_stats()

    def plot_pca(self):
        self.master.graph_frame.pca_plot(model=self.master.model, columns=self.master.columns)
        self.plot_stats()

    def plot_stats(self):
        self.master.stats_frame.plot()

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
        else:
            self.master.models[self.model_choice.get()]["params"][param]["value"] = self.master.models[self.model_choice.get()]["params"][param]["default"]


class StatsFrame(ctk.CTkScrollableFrame):

    def __init__(self, master=None):
        super().__init__(master, orientation="horizontal", width=master.horizontal_resolution-1600)

    def plot(self):
        
        if hasattr(self, "image_label"):
            self.image_label.destroy()

        values = [value for value, checkbox in self.master.master.master.model_choice_frame.value_checkboxes.items() if checkbox.get()]
        plot = self.master.master.master.clusterator.plot_features_distributiony_by_clusters(values)

        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)

        # Load the image from the buffer
        image = Image.open(buf)
        height = self.master.master.master.vertical_resolution
        width = image.width * height // image.height
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Create a CTkLabel to display the image
        self.image_label = ctk.CTkLabel(master=self, image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection
        self.image_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

class SaveFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        self.create_widgets()

    def create_widgets(self):
        pass


class GraphFrame(ctk.CTkFrame):

    def __init__(self, master=None):
        super().__init__(master)
        

    def plot(self):
        # remove previous image if exists
        if hasattr(self, "image_label"):
            self.image_label.destroy()

        values = [value for value, checkbox in self.master.model_choice_frame.value_checkboxes.items() if checkbox.get()]
        plot = self.master.clusterator.plot(show_values=values)

        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)

        # Load the image from the buffer
        image = Image.open(buf)
        height = self.master.vertical_resolution // 2
        width = height // 2 * 3
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Create a CTkLabel to display the image
        self.image_label = ctk.CTkLabel(master=self, image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection
        self.image_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

    def pca_plot(self, model, columns):

        if hasattr(self, "image_label"):
            self.image_label.destroy()
            
        values = [value for value, checkbox in self.master.model_choice_frame.value_checkboxes.items() if checkbox.get()]
        print(type(self.master))
        n_components = int(self.master.model_choice_frame.n_components_entry.get())
        plot = self.master.clusterator.pca_plot(model=model, columns=columns, n_components=n_components)

        buf = io.BytesIO()
        plot.savefig(buf, format='png')
        buf.seek(0)

        # Load the image from the buffer
        image = Image.open(buf)
        height = self.master.vertical_resolution // 2
        width = height // 2 * 3
        image = image.resize((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Create a CTkLabel to display the image
        self.image_label = ctk.CTkLabel(master=self, image=photo)
        self.image_label.image = photo  # Keep a reference to avoid garbage collection
        self.image_label.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

if __name__ == "__main__":
    app = App()
    app.run()
    app.save_models()
