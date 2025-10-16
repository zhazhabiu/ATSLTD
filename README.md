# ATSLTD
This is an unofficial implementation of 2019 ACM MM《Asynchronous Tracking-by-Detection on Adaptive Time Surfaces for Event-based Object Tracking》using Python.

## Model
*'model.yml.gz'* is from https://github.com/opencv/opencv_extra/blob/master/testdata/cv/ximgproc/model.yml.gz

## Prerequisites
python >= 3.0  
opencv-contrib-python (neccessary)  
numpy  
pandas  
openCV  

## Running dvs_meanshift
```Python
python main.py  
```

## Data Preparation
Test data should be put in *'./dataset/{name_of_dataset}/events.txt'*, or you can change the file reading path in *src/main.py*.


## Tracking results
All the trajectories will be saved in *'./{name_of_dataset}_tracking_res'*.
