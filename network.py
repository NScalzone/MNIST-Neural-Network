from typing import List

class NeuralNetwork:
    def __init__(self,
                 hidden_layer:List[float],
                 output_layer:List[float],
                 ):
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer
        
        
        
    def train_one_epoch(self, training_data:List, hidden_layer:List, output_layer:List)->None:
        pass