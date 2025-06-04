import json


with open('enhanced_products.json', 'r') as f:
    data = json.load(f)

for product in data:
    product['weightage'] = 1

with open('enhanced_products.json', 'w') as f:
    json.dump(data, f, indent=4)