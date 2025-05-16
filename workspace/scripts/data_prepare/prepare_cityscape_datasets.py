import pickle
import csv
import os
import sys
sys.path.append('../..')
# Importing Image class from PIL module
from PIL import Image
import random

def parse_csv(file_name):
    detection_record = {}
    file = open(file_name, 'r')
    reader = csv.reader(file)
    
    id_2_bounds = {}
    id_2_label_id = {}
    id_2_imgpath = {}

    # the columns are: image_id, img_path, label_id, xmin, ymin, xmax, ymax
    for row in reader:
        image_id = int(row[0])
        image_path = row[1]
        label_id = int(row[2])
        detection_record[image_id] = row

        xmin = int(float(row[3]))
        ymin = int(float(row[4]))
        xmax = int(float(row[5]))
        ymax = int(float(row[6]))
        id_2_bounds[image_id] = (xmin, ymin, xmax, ymax)
        id_2_label_id[image_id] = label_id
        id_2_imgpath[image_id] = image_path

    file.close()
    return detection_record, id_2_bounds, id_2_label_id, id_2_imgpath

def crop_image(cities):
    data_dir = "../../../benchmark/dataset/data/cityscape"
    for city_name in cities:
        # Define the path to the cityscape dataset
        cityscape_root = f'{data_dir}/leftImg8bit/train/{city_name}'
        cityscape_img_dir = os.path.join(f'{data_dir}/cropped', city_name)
        # create cityscape_img_dir if it does not exist
        if not os.path.exists(cityscape_img_dir):
            os.makedirs(cityscape_img_dir)

        # Define the path to the detection record
        detection_record_file = f'{data_dir}/sample_lists/citywise/{city_name}_fine.csv'

        # Load the detection record
        detection_record, id_2_bounds, id_2_label_id, id_2_imgpath = parse_csv(detection_record_file)

        # Prepare the cityscape dataset
        for image_id in id_2_imgpath.keys():
            img_path = os.path.join(cityscape_root, id_2_imgpath[image_id].split('/')[-1])
            img = Image.open(img_path)
            img_crop = img.crop(id_2_bounds[image_id])
            img_crop.save(os.path.join(cityscape_img_dir, f'{image_id}.png'))
            label_id = id_2_label_id[image_id]
            print(f'Processed {image_id} with label {label_id}')
        print(f'Finished preparing the cityscape dataset for {city_name}!')


if __name__ == '__main__':
    cities = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf', 
              'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach', 
              'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    crop_image(cities)
   