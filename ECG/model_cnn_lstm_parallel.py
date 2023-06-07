from pipeline import TrainTestPipeline
from keras import Model
from keras.layers import Input, Dense, Conv1D, LSTM, concatenate, GlobalAveragePooling1D
from keras.optimizers import Adam

class CNNLSTMParallelModelPipeline(TrainTestPipeline):
    '''
    Training Pipeline for CNN-LSTM in Parallel model
    '''   
    
    model_name = "CNNLSTMParallel"

    def create_model(self, input_size, output_size,
                     lstm_units,
                     conv_layers, conv_filters, conv_kernel_size, 
                     activation_function, learning_rate):
        
        input = Input(shape=input_size)

        cnn = Conv1D(conv_filters, (conv_kernel_size), 
                     activation=activation_function, padding="same")(input)
        for _ in range(conv_layers):
            cnn = Conv1D(conv_filters, (conv_kernel_size), 
                         activation=activation_function, padding="same")(cnn)
        cnn = GlobalAveragePooling1D()(cnn)

        lstm = LSTM(lstm_units)(input)

        x = concatenate([cnn, lstm])

        output = Dense(output_size, activation='softmax')(x)

        model = Model(input, output)

        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                    loss='categorical_crossentropy',
                    metrics = ['accuracy'])
                    
        return model
    
    def build_model(self, hp):
        input_size = (187, 1)
        output_size = 5

        # conv_layers = 5
        # conv_filters = 64
        # conv_kernel_size = 10
        # lstm_units = 80

        activation_function = 'relu'
        learning_rate = 0.001
        
        # hyperparameter search
        conv_layers = hp.Int("conv_layers", 3, 5)
        conv_filters = hp.Int("conv_filters", 48, 80, step=16)
        conv_kernel_size = hp.Int("conv_kernel_size", 4, 10, step=3)
        lstm_units = hp.Int("lstm_units", 48, 80, step=16)

        model = self.create_model(input_size, output_size,
                                  lstm_units,
                                  conv_layers, conv_filters, conv_kernel_size, 
                                  activation_function, learning_rate)
        
        return model


if __name__ == "__main__":
    pipeline = CNNLSTMParallelModelPipeline()
    pipeline.run()
     