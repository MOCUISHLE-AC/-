import json


with open("data/data_aishell/labels.json") as f:
    labels = json.load(f)
labels = dict([(labels[i], i) for i in range(len(labels))])
print(len(labels), labels)

