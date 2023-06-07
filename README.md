# ECG Arrhythmia Detection

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
