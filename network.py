from typing import List
import numpy as np
import math

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
        
        
    def train_one_epoch(self, training_data, learning_rate:int)->None:
        print(f"Training data length: {len(training_data)}")
        # total_inputs = len(training_data)
        total_inputs = 1
        run = 0
        while run < total_inputs:
            target_val = int(training_data[run][0])
            mnist_val = training_data[run][1:]
            # call forward step to step through the network for the current datapoint in the training set:
            self.forward_step(mnist_val)
            print(f"Output from step: {self.output}")
            
            predicted_value = evaluate_results(self.output)
            print(f"Target value: {target_val}, Predicted: {predicted_value}")
            
            print(f"Output weights pre-adjustment: {self.output_layer_weights}")
            self.update_weights(target_val, predicted_value, learning_rate)
            print(f"Output weights post-adjustment: {self.output_layer_weights}")
            
            run += 1
    
    def forward_step(self, inputs:List[float])->None:
        #set bias
        self.hidden_layer_outputs[0] = 1
        
        for i in range(len(self.hidden_layer_outputs)-1):
            self.hidden_layer_outputs[i+1] = calculate_output(inputs, self.hidden_layer_weights[i])
        
        for j in range(len(self.output)):
            self.output[j] = calculate_output(self.hidden_layer_outputs, self.output_layer_weights[j])
            
    def update_weights(self, target_value:int, predicted_value:int, learning_rate:int):
        for i in range(len(self.output)):
            tk = 0.1
            if i == target_value:
                if target_value == predicted_value:
                    tk = 0.9
            else:
                if i != predicted_value:
                    tk = 0.9
            output_error = calculate_output_error(self.output[i], tk)
            for j in range(len(self.output_layer_weights[i])):
                self.output_layer_weights[i][j] = adjust_weight(learning_rate, output_error, self.output_layer_weights[i][j])
            

def calculate_output(input_vector:List[float], weight_vector:List[float])->float: 
    dot_product = np.dot(input_vector, weight_vector)
    sigmoid = 1 / (1 + math.exp(-1 * dot_product))
    return sigmoid

def evaluate_results(results:List[float])->int:
    max_val = max(results)
    predicted_value = results.index(max_val)
    return predicted_value

def calculate_output_error(ok:float, tk:float)->float:
    error = ok * (1-ok) * (tk - ok)
    return error

def calculate_hidden_error():
    pass

def adjust_weight(learning_rate:int, error:float, current_weight_value:float)->float:
    weight = current_weight_value + (learning_rate * error * current_weight_value)
    return weight

    

