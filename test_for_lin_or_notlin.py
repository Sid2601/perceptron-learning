
#Problem: To test your data either linear or not linear with the help of single perceptron learning
import numpy as np
from perceptron import Perceptron

# i made dataset of 4 inpputs 
# 1) cough
# 2) cold
# 3) went to out of india
# 4) meet any corona patient in last 15 days

# and the result will be chances of patient  either corona positive or negative 
training_inputs = []

training_inputs.append(np.array([0,0,0,0])) 
training_inputs.append(np.array([1,0,0,0]))
training_inputs.append(np.array([1,1,0,0]))
training_inputs.append(np.array([0,1,0,0]))
training_inputs.append(np.array([0,0,1,0]))
training_inputs.append(np.array([1,1,1,0]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([1,0,1,0]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([0,1,1,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([1,0,1,1]))
training_inputs.append(np.array([0,0,1,1]))
training_inputs.append(np.array([1,1,0,1]))
training_inputs.append(np.array([0,0,0,1]))
training_inputs.append(np.array([0,0,1,0]))
training_inputs.append(np.array([0,1,0,1]))
training_inputs.append(np.array([1,1,1,1]))
training_inputs.append(np.array([1,0,0,1]))


labels = np.array([0,0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1])

perceptron = Perceptron(4)
perceptron.train(training_inputs,labels)


inputs = np.array([1,1,1,1])
print("epected output --->" +str(1))
print("perceptron output is ---> {}".format(perceptron.predict(inputs)))


print("---------------------------")

inputs = np.array([0,1,1,0])
print("epected output --->" +str(0))
print("perceptron output is ---> {}".format(perceptron.predict(inputs)))

print("---------------------------")
inputs = np.array([1,0,0,1])
print("epected output --->" +str(1))
print("perceptron output is ---> {}".format(perceptron.predict(inputs)))

print("final weights --> {}".format(perceptron.weights))

