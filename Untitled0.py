#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

path = "./car_logo_finail/Car logo/trainImg"  # 資料夾路徑
prefix = "0."  # 檔名前綴
suffix = ".jpg" or ".JPG"# 檔名後綴
start = 100  # 起始編號
end = 499  # 結束編號

# 讀取資料夾中的圖片檔名，存放在 filenames 中
filenames = os.listdir(path)

# 將檔名轉換為數字形式，存放在 indices 中
indices = [int(filename.replace(prefix, "").replace(suffix, "")) for filename in filenames]

# 創建預期存在的圖片編號列表
expected_indices = list(range(start, end+1))

# 找出缺失的圖片編號
missing_indices = sorted(set(expected_indices) - set(indices))

# 印出結果
if len(missing_indices) == 0:
    print("所有圖片都存在")
else:
    print("缺失的圖片編號：")
    print(missing_indices)

