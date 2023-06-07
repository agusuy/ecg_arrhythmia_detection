import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from tensorflow import argmax


'''
This file contains useful functiond for the project
'''


def plot_training_metrics(history, validation=True, save_path=None):
    '''
    Show training metrics over time from the history training result object
    '''

    # display accuracy
    plt.figure(0)
    plt.plot(history.history['accuracy'])
    if validation:
        plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train', 'validation'], loc='upper left')
    if save_path:
        plot_name = "training_accuracy.png"
        plot_filepath = os.path.join(save_path, plot_name)
        plt.savefig(plot_filepath)
    else:
        plt.show()
    
    # display loss
    plt.figure(1)
    plt.plot(history.history['loss'])
    if validation:
        plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    if validation:
        plt.legend(['train', 'validation'], loc='upper left')
    if save_path:
        plot_name = "training_loss.png"
        plot_filepath = os.path.join(save_path, plot_name)
        plt.savefig(plot_filepath)
    else:
        plt.show()

def evaluate_model(model, x_test, y_test, labels, save_path=None):
    '''
    Evaluate model using:
        Accuracy, Loss, F1, and Confusion Matrix
    '''

    predictions = model.predict(x_test)
    label_predicted = argmax(predictions, axis=1)
    label_expected = argmax(y_test, axis=1)

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    print('Test Loss:', round(loss, 3))
    print('Test Accuracy:', round(accuracy, 3))

    f1 = f1_score(label_expected, label_predicted, average='weighted')
    print('Test F1:', round(f1, 3))



    cm = confusion_matrix(label_expected, label_predicted, normalize='true')
    cm_plot = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels=labels)
    cm_plot.plot()
    if save_path:
        plot_name = "confusion_matrix.png"
        plot_filepath = os.path.join(save_path, plot_name)
        plt.savefig(plot_filepath)

        log = "Test Loss:" + str(round(loss, 3)) + "\n"
        log += "Test Accuracy:" + str(round(accuracy, 3)) + "\n"
        log += "Test F1:" + str(round(f1, 3)) + "\n"
        write_log(save_path, log)
    else:
        plt.show()

def write_log(path, text):
    '''
    Function to write results in a log file
    '''
    log_file = os.path.join(path, "log.txt")
    f = open(log_file, "a")
    f.write(text)
    f.close()
