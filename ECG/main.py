from model_cnn import *
from model_lstm import *
from model_cnn_lstm_parallel import *
from model_cnn_lstm_sequential import *

# Run all pipelines

CNNModelPipeline().run()
LSTMModelPipeline().run()
CNNLSTMParallelModelPipeline().run()
CNNLSTMSequentialModelPipeline().run()
