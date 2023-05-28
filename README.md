# 1112-machine-learning   
car-logo-finail   
NCNU 1112 machine learning midterm project
## Motivation
There are various brands of cars, and different car brands will also make unique signs belonging to their own car factories to distinguish whether they are their own cars. After taking this course this semester, I realized that computers have different learning modes. Through the mid-term project, we designed a neural network-like model to identify car logos, and we chose high-end car brands. 
## Data Collection 
We use one smartphone and one monocular camera to collect cars logo   
We choose ten car showrooms to collect our photo and shoot 100 photos for each logo from different angles.
## Environment
Cuda 11.2    
Numpy 1.23.5     
Pandas 2.0.0     
Tensorflow 2.12.0    
Tqdm 4.65.0     
Opencv 4.7.0.72     
Kreas 2.12.0    
## Preproocessing
Augmentation     
   90 degree     
   180 degree    
   270 degree    
Has total 4100 pictures    
    4000 pictures for training     
    100 for test     
## CNN model
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/5c9c45f8-02ec-4169-b82c-bf5953fa1f2c)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/e6316343-fe95-44a2-a1e8-3092a9b4d967)
#### result
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/068f4e05-df40-4ec0-926a-cd5a0468798e)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/c84009f3-5b4c-4a0a-89a8-36ef5fb6ec89)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/c5e11f16-c409-44d5-8a16-f2f7e0a6c174)
## VGG16
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/31b7072a-698b-423e-98b3-1519c9bacfea)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/1e6f1a5f-bbb6-49e9-9616-21b117cc1792)
#### result
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/4889b6e1-a538-4ce5-9e67-8772d4c91a3b)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/77a4a69d-4b93-4646-8331-90c3886e4e48)
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/7e6afb04-9b68-4c6b-9ab3-45ec64c2b547)

