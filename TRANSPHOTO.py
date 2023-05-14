#!/usr/bin/env python
# coding: utf-8

# In[20]:


import os
import shutil

# 資料夾路徑
input_dir = './car_logo_finail/Car logo/trainImg/volkswagen'
output_dir = './car_logo_finail/Car logo/trainImg'

# 從001開始編號
count = 100

# 對資料夾中的每一個檔案進行處理
for filename in os.listdir(input_dir):
    # 確定檔案是圖片檔
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.JPG'):
        # 新檔案名稱
        new_filename = '9.' + str(count) + filename[-4:]
        # 建立複製路徑
        output_path = os.path.join(output_dir, new_filename)
        # 複製檔案並重新命名
        shutil.copy(os.path.join(input_dir, filename), output_path)
        # 更新編號
        count += 1


# In[21]:


import os
import shutil

# 資料夾路徑
input_dir = './car_logo_finail/Car logo/testImg/volkswagen'
output_dir = './car_logo_finail/Car logo/testImg'

# 從001開始編號
count = 0

# 對資料夾中的每一個檔案進行處理
for filename in os.listdir(input_dir):
    # 確定檔案是圖片檔
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png') or filename.endswith('.JPG'):
        # 新檔案名稱
        new_filename = '9.' + str(count) + filename[-4:]
        # 建立複製路徑
        output_path = os.path.join(output_dir, new_filename)
        # 複製檔案並重新命名
        shutil.copy(os.path.join(input_dir, filename), output_path)
        # 更新編號
        count += 1

