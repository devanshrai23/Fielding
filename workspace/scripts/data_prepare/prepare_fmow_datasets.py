import os
import json
import pickle
import sys
sys.path.append("../../")
import matplotlib.pyplot as plt
import datetime
import csv
import random
import numpy as np
# Importing Image class from PIL module
from PIL import Image

"""
Go through the samples to be used, read the bounding box information from the metadata json file,
crop the image based on the bounding box, resize to 224*224, and save the cropped image
"""
def preprocess_fmow_samples(country_samples):
    intput_dir = None # default to "/data/fMoW/train", replace it if needed
    output_dir = None # e.g.: "/data/fMoW/train_cropped"
    if intput_dir is None:
        print("FMoW dataset input_dir is None, set it to /data/fMoW/train by default")
        intput_dir = "/data/fMoW/train"
    if output_dir is None:
        print("FMoW dataset output_dir is None, please set it")
        return
    # create output_dir if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for country_code, samples in country_samples.items():
        print(f"Processing {len(samples)} samples for country {country_code}")
        for sample in samples:
            img_path = intput_dir+sample[0][16:]
            metadata = sample[1]
            class_name = metadata["class"]
            if os.path.exists(os.path.join(output_dir, f"{country_code}_{class_name}_{os.path.basename(img_path)}")):
                print(f"Sample {country_code}_{os.path.basename(sample[0])} already exists, skip")
                continue
            box = metadata["box"]
            # four box values can be thought of as corresponding to [left, top, width, height] 
            # for a box surrounding the object of interest.
            left, top, width, height = box
            img = Image.open(img_path)
            img = img.crop((left, top, left + width, top + height))
            img = img.resize((224, 224))
            img.save(os.path.join(output_dir, f"{country_code}_{class_name}_{os.path.basename(img_path)}"))
        print(f"Country {country_code} samples processed")

############################################################################################################

with open("country_zone_samples_in_2015-2018.json", "r") as f:
    country_samples = json.load(f)

preprocess_fmow_samples(country_samples=country_samples)
