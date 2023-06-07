from pipeline import TrainTestPipeline
from keras.models import Sequential
from keras.layers import InputLayer, Dense, Conv1D, LSTM, MaxPool1D
from keras.optimizers import Adam

class CNNLSTMSequentialModelPipeline(TrainTestPipeline):
    '''
    Training Pipeline for CNN LSTM Sequential model
    '''

    model_name = "CNNLSTMSequential"

    def create_model(self, input_size, output_size,
                     conv_blocks, conv_filters, conv_kernel_size, pooling_size, 
                     lstm_units,
                     fc_layers, fc_units,
                     activation_function, learning_rate):
        
        model = Sequential()
        
        model.add(InputLayer(input_shape=input_size))
        
        for _ in range(conv_blocks):
            model.add(Conv1D(conv_filters, (conv_kernel_size), 
                             activation=activation_function, padding="same"))
            model.add(MaxPool1D(pool_size=pooling_size, padding="same"))

        model.add(LSTM(lstm_units))
        
        for _ in range(fc_layers):
            model.add(Dense(fc_units, activation=activation_function))
        
        model.add(Dense(output_size, activation='softmax'))

        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                    loss='categorical_crossentropy',
                    metrics = ['accuracy'])
                    
        return model
    
    def build_model(self, hp):
        input_size = (187, 1)
        output_size = 5

        # conv_blocks = 5
        # conv_filters = 64
        # conv_kernel_size = 10
        # pooling_size = 3
        # lstm_units = 80
        # fc_layers = 1
        # fc_units = 8

        activation_function = 'relu'
        learning_rate = 0.001
        
        # hyperparameter search
        conv_blocks = hp.Int("conv_blocks", 3, 5)
        conv_filters = hp.Int("conv_filters", 48, 80, step=16)
        conv_kernel_size = hp.Int("conv_kernel_size", 4, 10, step=3)
        pooling_size = hp.Int("pooling_size", 2, 3)
        lstm_units = hp.Int("lstm_units", 48, 80, step=16)
        fc_layers = hp.Int("fc_layers", 1, 3)
        fc_units = hp.Int("fc_units", 8, 16, step=8)

        model = self.create_model(input_size, output_size,
                     conv_blocks, conv_filters, conv_kernel_size, pooling_size, 
                     lstm_units,
                     fc_layers, fc_units,
                     activation_function, learning_rate)
        
        return model


if __name__ == "__main__":
    pipeline = CNNLSTMSequentialModelPipeline()
    pipeline.run()
     