import json

with open("data/backup_models.json", "r") as file:
    models = json.load(file)

for model, details in models.items():
    for param in details["params"]:
        details["params"][param]["default"] = details["params"][param]["value"]
        # print(param, details["params"][param])

with open("data/backup_models.json", "w") as file:
    models = json.dump(models, file, indent=4)