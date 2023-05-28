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
![image](https://github.com/Rayray7777777/1112-machine-learning/assets/132484524/a13fef7e-d3f4-452e-8677-41273ea46579)
