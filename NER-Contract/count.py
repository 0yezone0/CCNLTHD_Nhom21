import json

with open("train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(len(data))
