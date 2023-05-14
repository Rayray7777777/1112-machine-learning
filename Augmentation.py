#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cv2,os
from tqdm import tqdm

dir_1 = './汽車logo照片/ferrari/法拉利/'
dir_2 = './car_logo_finail/Car logo/testImg/bentley/'

b = []

for name in os.listdir(dir_1):
    b.append(name)

# Rotate image
def rotate(path):
    for n in tqdm(b):
        p  = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            img_90 = p + 'aug_90_' + i
            img_180 = p + 'aug_180_' + i
            img_270 = p + 'aug_270_' + i
            s = cv2.imread(img)
            r90 = cv2.rotate(s,cv2.ROTATE_90_CLOCKWISE)
            r180 = cv2.rotate(s,cv2.ROTATE_180)
            r270 = cv2.rotate(s,cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(img_90,r90)
            cv2.imwrite(img_180,r180)
            cv2.imwrite(img_270,r270)

# Resize image
def resize(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.resize(s,(224,224))
            cv2.imwrite(img,s)

def resize_images_in_dir(dir_path, size=(224, 224)):
    for root, dirs, files in os.walk(dir_path):
        for filename in tqdm(files):
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, size)
            cv2.imwrite(file_path, img)



# Gray scale
def convert_gray(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.cvtColor(s,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(img,s)


resize_images_in_dir(dir_1)
#resize_images_in_dir(dir_2)


# In[ ]:


import cv2,os
from tqdm import tqdm

dir_1 = './car_logo_finail/Car logo/trainImg/'
#dir_2 = '/content/drive/MyDrive/Colab Notebooks/car_logo_finail/Car logo/testImg/volkswagen/'

b = []

for name in os.listdir(dir_1):
    b.append(name)

# Rotate image
def rotate(path):
    for n in tqdm(b):
        p  = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            img_90 = p + 'aug_90_' + i
            img_180 = p + 'aug_180_' + i
            img_270 = p + 'aug_270_' + i
            s = cv2.imread(img)
            r90 = cv2.rotate(s,cv2.ROTATE_90_CLOCKWISE)
            r180 = cv2.rotate(s,cv2.ROTATE_180)
            r270 = cv2.rotate(s,cv2.ROTATE_90_COUNTERCLOCKWISE)

            cv2.imwrite(img_90,r90)
            cv2.imwrite(img_180,r180)
            cv2.imwrite(img_270,r270)

# Resize image
def resize(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.resize(s,(224,224))
            cv2.imwrite(img,s)

def resize_images_in_dir(dir_path, size=(224, 224)):
    for root, dirs, files in os.walk(dir_path):
        for filename in tqdm(files):
            file_path = os.path.join(root, filename)
            img = cv2.imread(file_path)
            img = cv2.resize(img, size)
            cv2.imwrite(file_path, img)



# Gray scale
def convert_gray(path):
    for n in tqdm(b):
        p = path + n + '/'
        for i in os.listdir(p):
            img = p + i
            s = cv2.imread(img)
            s = cv2.cvtColor(s,cv2.COLOR_RGB2GRAY)
            cv2.imwrite(img,s)

rotate(dir_1)

