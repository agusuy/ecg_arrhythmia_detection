import io
import os
from abc import ABC, abstractmethod
from keras.models import load_model
from keras_tuner import GridSearch
from project_constants import RESULTS_FOLDER
from ecg_dataset import load_ecg_data
from utils import evaluate_model, plot_training_metrics, write_log


class TrainTestPipeline(ABC):
    '''
    Abstract class to create a model training pipeline
    '''

    @property
    @abstractmethod
    def model_name(self):
        '''
        Class property with models name
        '''
        pass

    def __init__(self):
        self.model_folder = os.path.join(RESULTS_FOLDER, self.model_name)
        self.model_filepath = os.path.join(self.model_folder, self.model_name+".h5")

        try:
            # create model folder to store results
            os.makedirs(self.model_folder)
        except OSError:
            # folder already exists
            
            log_file = os.path.join(self.model_folder, "log.txt")
            if os.path.exists(log_file):
                # remove a previous log file
                os.remove(log_file)
        

    @abstractmethod
    def create_model():
        '''
        Abstract method for creating the model from hyperparameters
        '''
        pass

    @abstractmethod
    def build_model(self, hp):
        '''
        Abstract method for defining the hyperparameter search
        '''
        pass
    
    def run(self):
        '''
        Method to run the pipeline
        '''
        print("-------------"+self.model_name+"-------------")
        try:
            
            # load dataset
            data = load_ecg_data(balance_data=True)
            labels, x_train, _, y_train_encoded, x_test, _, y_test_encoded = data

            try:
                # Try to load an existing model
                model = load_model(self.model_filepath)
                print("Loading existing model")

            except IOError:
                # If model does not exist create and train a new one

                print("Creating and training the model")
                ############################################################
                print("Hyperparameter tuning")

                # Performs hyperparameter search with Grid Search
                tuner = GridSearch(
                    hypermodel=self.build_model,
                    objective="val_accuracy",
                    overwrite=True,
                    directory="Trials",
                    project_name=self.model_name,
                    max_consecutive_failed_trials=10
                )

                epochs_search=5
                tuner.search(x_train, y_train_encoded,
                            validation_data=(x_test,y_test_encoded),
                            epochs=epochs_search)
                
                # Get the best set of hyperparameters found
                best_hps = tuner.get_best_hyperparameters(1)
                print("best hyperparameters:", best_hps[0].values)
                write_log(self.model_folder, "best hyperparameters: " + str(best_hps[0].values) + "\n")
                model = self.build_model(best_hps[0])
                ############################################################
                print("Re train best model")

                # Train the final model
                epochs = 20
                batch_size = 32

                history = model.fit(x_train, y_train_encoded, 
                                    epochs=epochs, batch_size=batch_size, 
                                    validation_data=(x_test,y_test_encoded))

                plot_training_metrics(history, save_path=self.model_folder)

                model.save(self.model_filepath, save_format="h5")

            finally:
                # Evaluate model

                stream = io.StringIO()
                model.summary(print_fn=lambda x: stream.write(x + '\n'))
                summary_string = stream.getvalue()
                stream.close()
                write_log(self.model_folder, "model architecture:" + summary_string + "\n")

                print("Evaluating model")
                evaluate_model(model, x_test, y_test_encoded, labels, save_path=self.model_folder)
            
        except Exception as e:
            print(self.model_name, "ended with error")
