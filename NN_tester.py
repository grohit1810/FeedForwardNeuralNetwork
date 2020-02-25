import Ann_tanh as ann_tanh
import numpy as np
import pickle
import random
data=pickle.load(open("data.pkl","rb"))
nn=ann_tanh.NeuralNetwork(2,4,2)
inputs=[]
targets=[]

training_size = int(0.8*len(data))
validation_size = int(0.1*len(data))
num_epocs = 300

for i in xrange(0,len(data)):
    targets.append(data[i]['out'])
    inputs.append(data[i]['inp'])
nn.train(inputs[:training_size], targets[:training_size], (inputs[training_size:training_size+validation_size], targets[training_size:training_size+validation_size]), num_epocs)
c=0
for j in xrange(0,50):
    
    s=nn.predict(data[j+training_size+validation_size]['inp'])
    s1=data[j+training_size+validation_size]['out']
    #print "Original: ",data[j+450]['out'], "Predicted: ", s, "Input: ",data[j+450]['inp']  
    if s == s1:
        c+=1
#print j
#print "Correctly identified test cases : ",c
print "Accuracy : ",((c*1.0)/(j+1))*100

