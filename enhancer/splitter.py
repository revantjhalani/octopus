import json 
with open("enhanced_products.json", "r") as f:
    data = json.load(f)

import os
import slugify
os.makedirs("products", exist_ok=True)


def preprocess_products(data):
    """Preprocess products to add weight-based scoring and enhance searchability"""
    processed_data = []
    
    for product in data:
        product['_weight'] = 1  # Default weight of 1 if not specified
        
        processed_data.append(product)
    
    return processed_data

data = preprocess_products(data)

for item in data:
    with open(f"products/{str(item['id']) + "-" + slugify.slugify(item['name'])}.json", "w") as f:
        json.dump(item, f, indent=2)




    