from pipeline import TrainTestPipeline
from keras.models import Sequential
from keras.layers import InputLayer, Dense, LSTM, MaxPool1D, GlobalAveragePooling1D
from keras.optimizers import Adam

class LSTMModelPipeline(TrainTestPipeline):
    '''
    Training Pipeline for LSTM model
    '''
        
    model_name = "LSTM"

    def create_model(self, input_size, output_size,
                     lstm_layers, lstm_units,
                     fc_layers, fc_units,
                     activation_function, learning_rate):

        model = Sequential()
        
        model.add(InputLayer(input_shape=input_size))

        for _ in range(lstm_layers):
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

        # lstm_units = 64
        # fc_layers = 1
        # fc_units = 16

        lstm_layers = 1
        activation_function = 'relu'
        learning_rate = 0.001
        
        # hyperparameter search
        lstm_units = hp.Int("lstm_units", 48, 80, step=16)
        fc_layers = hp.Int("fc_layers", 1, 3)
        fc_units = hp.Int("fc_units", 8, 16, step=8)

        model = self.create_model(input_size, output_size, 
                                  lstm_layers, lstm_units,
                                  fc_layers, fc_units, 
                                  activation_function, learning_rate)
        
        return model


if __name__ == "__main__":
    pipeline = LSTMModelPipeline()
    pipeline.run()
     