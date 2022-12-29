import pandas as pd
import os
from classes import classes
from random import random
from shutil import copyfile
from shutil import rmtree
from PIL import Image

data = pd.read_csv("metadata.csv")
for path in ['train', 'test', 'val']:
    for cls in classes:
        try:
            os.makedirs(f'sorted_skin_cancer/{path}/{cls}')
        except OSError:
            rmtree(f'sorted_skin_cancer/{path}/{cls}')
            os.makedirs(f'sorted_skin_cancer/{path}/{cls}')


images_path = 'Skin Cancer'
sorted_path = 'sorted_skin_cancer'
for i, row in data.iterrows():
    prob = random()
    if prob < 0.7:
        sub_path = 'train'
    elif prob < 0.9:
        sub_path = 'test'
    else:
        sub_path = 'val'
    original_path = f'{images_path}/{row["image_id"]}.jpg'
    if row["dx"] != "nv":
        Original_Image = Image.open(original_path)
        Original_Image = Original_Image.resize((224, 224))
        rotated_image1 = Original_Image.rotate(90)
        rotated_image2 = rotated_image1.rotate(90)
        rotated_image3 = rotated_image2.rotate(90)

        Original_Image = Original_Image.save(f'{sorted_path}/{sub_path}/{row["dx"]}/{row["image_id"]}_0.jpg')
        rotated_image1 = rotated_image1.save(f'{sorted_path}/{sub_path}/{row["dx"]}/{row["image_id"]}_1.jpg')
        rotated_image2 = rotated_image2.save(f'{sorted_path}/{sub_path}/{row["dx"]}/{row["image_id"]}_2.jpg')
        rotated_image3 = rotated_image3.save(f'{sorted_path}/{sub_path}/{row["dx"]}/{row["image_id"]}_3.jpg')
    else:
        copyfile(original_path, f'{sorted_path}/{sub_path}/{row["dx"]}/{row["image_id"]}.jpg')
    if (i+1) % 500 == 0:
        print(f'Creating random datasets from original large imageset, {i+1}/10014')



