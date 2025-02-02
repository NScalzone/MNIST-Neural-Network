from data_processing import generate_weights,get_data
from network import NeuralNetwork
from tqdm import tqdm
import matplotlib.pyplot as plt
import csv


HIDDEN_UNITS = 20
EPOCHS = 3
LEARNING_RATE = 0.1
MOMENTUM = 0.9
TRAINING_DATA_PATH = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/mnist_train_with_bias.csv"
TESTING_DATA_PATH = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/mnist_test_with_bias.csv"
HIDDEN_SAVE_PATH = f"/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/hidden_layer_weights_{HIDDEN_UNITS}_units.csv"
OUTPUT_SAVE_PATH = f"/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/output_layer_weights_{HIDDEN_UNITS}_hidden_units.csv"

hidden_layer_weights = generate_weights(HIDDEN_UNITS, 785)
output_weights = generate_weights(10, (HIDDEN_UNITS + 1))
training_data = get_data(TRAINING_DATA_PATH)
testing_data = get_data(TESTING_DATA_PATH)

neuralnet = NeuralNetwork(hidden_layer_weights, output_weights, HIDDEN_UNITS)

epochs = []
traindata_test_accuracy = []
testdata_test_accuracy = []

for current_run in range(EPOCHS):
    print(f"Current training epoch: {current_run+1} out of {EPOCHS}")
    
    neuralnet.train_one_epoch(training_data, LEARNING_RATE, MOMENTUM)
    
    testdata_test_accuracy.append(neuralnet.run_test(testing_data))
    
    # traindata_test_accuracy.append(neuralnet.run_test(training_data))
    epochs.append(current_run + 1)
    

print(f"Training set results: {traindata_test_accuracy}\nTest set results: {testdata_test_accuracy}")

# plt.plot(epochs, traindata_test_accuracy, label="Training Data % Correct")
plt.plot(epochs, testdata_test_accuracy, label="Testing Data % Correct")
plt.legend()
plt.show()

with open(HIDDEN_SAVE_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(neuralnet.hidden_layer_weights)
    
with open(OUTPUT_SAVE_PATH, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(neuralnet.output_layer_weights)