from typing import List
import numpy as np
import math
from tqdm import tqdm

class NeuralNetwork:
    def __init__(self,
                 hidden_layer_weights:List[float],
                 output_layer_weights:List[float],
                 hidden_units:int,
                 ):
        self.hidden_layer_weights = hidden_layer_weights
        self.output_layer_weights = output_layer_weights
        self.hidden_layer_outputs = []
        for i in range(hidden_units + 1):
            self.hidden_layer_outputs.append(0)
        self.output = [0,0,0,0,0,0,0,0,0,0]
        
        self.hidden_layer_momentum_weights = []
        self.output_layer_momentum_weights = []
        for j in range(len(self.hidden_layer_weights)):
            self.hidden_layer_momentum_weights.append([])
            for k in range(len(self.hidden_layer_weights[j])):
                self.hidden_layer_momentum_weights[j].append(0)
        for l in range(len(self.output_layer_weights)):
            self.output_layer_momentum_weights.append([])
            for m in range(len(self.output_layer_weights[l])):
                self.output_layer_momentum_weights[l].append(0)
        
    def train_one_epoch(self, training_data, learning_rate:float, momentum:float)->None:
        print(f"Training")
        total_inputs = len(training_data)
        
        for run in tqdm(range(total_inputs)):
            target_val = int(training_data[run][0])
            mnist_val = training_data[run][1:]
            
            # call forward step to step through the network for the current datapoint in the training set:
            self.forward_step(mnist_val)
           
            # If predicted value is incorrect, adjust weights:
            predicted_value = evaluate_results(self.output)
            if predicted_value != target_val:
                self.update_weights(target_val, predicted_value, learning_rate, momentum, mnist_val)
        
    def run_test(self, test_data)->float:
        total_inputs = len(test_data)
        correct_answers = 0
        print("Running Tests")
        for run in tqdm(range(total_inputs)):
            target_val = int(test_data[run][0])
            mnist_val = test_data[run][1:]
            
            # call forward step to step through the network for the current datapoint in the training set:
            self.forward_step(mnist_val)
            
            predicted_value = evaluate_results(self.output)
            if predicted_value == target_val:
                correct_answers += 1
        
        percent_correct = 100 * (correct_answers/total_inputs)
        return percent_correct
    
    def create_confusion_matrix(self, test_data)->None:
        confusion_matrix = []
        for i in range(10):
            temp = []
            confusion_matrix.append(temp)
            for j in range(10):
                confusion_matrix[i].append(0)
        total_inputs = len(test_data)
     
   
        for run in range(total_inputs):
            target_val = int(test_data[run][0])
            mnist_val = test_data[run][1:]
            
            # call forward step to step through the network for the current datapoint in the training set:
            self.forward_step(mnist_val)            
            predicted_value = evaluate_results(self.output)
            confusion_matrix[int(target_val)][int(predicted_value)] += 1

        print("============================================================================================")
        print("Confusion Matrix:")
        print("Predicted Values:\t0\t1\t2\t3\t4\t5\t6\t7\t8\t9\trecall\n----------------------------------------------------------------------------------------------------")
        precision = [0,0,0,0,0,0,0,0,0,0]
        for k in range(10):
            print(f"Target value: {k}\t\t", end='')
            sum = 0
            for l in range(10):
                print(f"{confusion_matrix[k][l]}\t", end='')
                sum += confusion_matrix[k][l]
                precision[l] += confusion_matrix[k][l]
            recall = int(100 * ((confusion_matrix[k][k])/sum))
            print(recall)
            print('----------------------------------------------------------------------------------------------------')

        print("Precision:\t\t",end='')
        for m in range(10):
            precison_value = int(100*((confusion_matrix[m][m])/(precision[m])))
            print(f"{precison_value}\t",end='')
            
        print('\n----------------------------------------------------------------------------------------------------')
    
    def forward_step(self, inputs:List[float])->None:
        #set bias
        self.hidden_layer_outputs[0] = 1
        
        for i in range(len(self.hidden_layer_outputs)-1):
            self.hidden_layer_outputs[i+1] = calculate_output(inputs, self.hidden_layer_weights[i])
        
        for j in range(len(self.output)):
            self.output[j] = calculate_output(self.hidden_layer_outputs, self.output_layer_weights[j])
            
    def update_weights(self, target_value:int, predicted_value:int, learning_rate:float, momentum:float, training_inputs:List[float]):
        output_errors = []
        for i in range(len(self.output)):
            tk = 0.1
            if i == target_value:
                tk = 0.9
            
            output_error = calculate_output_error(self.output[i], tk)
            output_errors.append(output_error)
            for j in range(len(self.output_layer_weights[i])):
                original_weight = self.output_layer_weights[i][j]
                self.output_layer_weights[i][j] = adjust_weight(learning_rate, output_error, self.output_layer_weights[i][j],momentum,self.output_layer_momentum_weights[i][j], output_value=self.hidden_layer_outputs[j])
                self.output_layer_momentum_weights[i][j] = (self.output_layer_weights[i][j] - original_weight)
        
        for k in range(len(self.hidden_layer_outputs)-1):
            hj = self.hidden_layer_outputs[k+1]
            output_error_sum = 0
            for n in range(len(output_errors)):
                output_error_sum += (self.output_layer_weights[n][k] * output_errors[n])
            hidden_error = calculate_hidden_error(hj, output_error_sum)
            for m in range(len(self.hidden_layer_weights[k])):
                original_weight = self.hidden_layer_weights[k][m]
                self.hidden_layer_weights[k][m] = adjust_weight(learning_rate, hidden_error, self.hidden_layer_weights[k][m],momentum,self.hidden_layer_momentum_weights[k][m], training_inputs[m])
                self.hidden_layer_momentum_weights[k][m] = (self.hidden_layer_weights[k][m] - original_weight)

        
def calculate_output(input_vector:List[float], weight_vector:List[float])->float: 
    dot_product = np.dot(input_vector, weight_vector)
    try:
        sigmoid = 1 / (1 + math.exp(-1 * dot_product))
    except OverflowError:
        sigmoid = 1
    return sigmoid

def evaluate_results(results:List[float])->int:
    max_val = max(results)
    predicted_value = results.index(max_val)
    return predicted_value

def calculate_output_error(ok:float, tk:float)->float:
    error = ok * (1-ok) * (tk - ok)
    return error

def calculate_hidden_error(hj, output_error_sum):
    error = hj * (1-hj) * output_error_sum
    return error

def adjust_weight(learning_rate:int, error:float, current_weight_value:float, momentum:float, momentum_weight:float, output_value:float)->float:
    delta_w = learning_rate * error * output_value
    momentum_adjustment = momentum * momentum_weight
    weight = current_weight_value + delta_w + momentum_adjustment
    return weight

    

