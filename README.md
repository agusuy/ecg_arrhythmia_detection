# ECG Arrhythmia Detection

## Description
The automatic detection of heart arrhythmias from electrocardiogram (ECG) measurements using Artificial Intelligence techniques has the potential to benefit health care by preventing some of the most common causes of death worldwide. This research has the objective of testing existing solutions as well as new proposed solutions for the task of detecting irregular heartbeat rhythms.

This project proposes two novel deep learning models for detecting the presence of arrhythmias on ECGs which leverage a combination of Convolutional Neural Networks (CNN) and Long Short Term Memory Networks (LSTM) working together.



## Instructions
requirements.txt file has the python packages required by the project

Download dataset from this links:

-Train Dataset:
https://drive.google.com/file/d/1vcZkFUihTJlVlEdhDDHAry-63afzY864/view?usp=sharing

-Test Dataset:
https://drive.google.com/file/d/1OU-ChGwgW1IMAIX1Tpb7iqw8dHpFu4G7/view?usp=sharing

Put dataset files inside the "ECG/dataset/" folder

Each model has its own training pipeline that can be run independently:
python model_cnn.py
python model_lstm.py
python model_cnn_lstm_parallel.py
python model_cnn_lstm_sequential.py

There is an aditional python file to run all training pipelines
python main.py
