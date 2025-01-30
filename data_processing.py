import random
from typing import List

def generate_weights(units:int)->List[float]:
    model_weights = []
    for i in range(units):
        temp = []
        for j in range(5):
            # number = round((random.uniform(-0.5,0.5)),1)
            # number = round(number, 1)
            temp.append(round((random.uniform(-0.5,0.5)),1))
        model_weights.append(temp)

    return model_weights

weights = generate_weights(5)
print(weights)

