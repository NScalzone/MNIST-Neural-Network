from data_processing import generate_weights,get_data
from network import NeuralNetwork
from tqdm import tqdm

HIDDEN_UNITS = 20
EPOCHS = 5
LEARNING_RATE = 0.001
TRAINING_DATA_PATH = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/mnist_train_with_bias.csv"
TESTING_DATA_PATH = "/Users/nicholasscalzone/Documents/COMPUTER SCIENCE CLASSES/Machine Learning/MNIST-Neural-Network/mnist_test_with_bias.csv"



hidden_layer_weights = generate_weights(HIDDEN_UNITS, 785)
output_weights = generate_weights(10, (HIDDEN_UNITS + 1))
training_data = get_data(TRAINING_DATA_PATH)
testing_data = get_data(TESTING_DATA_PATH)

neuralnet = NeuralNetwork(hidden_layer_weights, output_weights, HIDDEN_UNITS)

traindata_test_accuracy = []
testdata_test_accuracy = []

for current_run in tqdm(range(EPOCHS)):
    
    neuralnet.train_one_epoch(training_data, LEARNING_RATE)
    
    testdata_test_accuracy.append(neuralnet.run_test(testing_data))
    
    traindata_test_accuracy.append(neuralnet.run_test(training_data))
    

print(f"Training set results: {traindata_test_accuracy}\nTest set results: {testdata_test_accuracy}")

