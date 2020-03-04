import Ann_enhanced_deep as ann
import numpy as np
import pickle
import random
cat_samples=pickle.load(open("cat_samples.pkl","rb"))
deer_samples=pickle.load(open("deer_samples.pkl","rb"))
nn=ann.NeuralNetwork(1024,256,32,2, learning_rate=0.01)
cat_train_samples = cat_samples['train']
deer_train_samples = deer_samples['train']

cat_test_samples = cat_samples['test']
deer_test_samples = deer_samples['test']
inputs=[]
targets=[]


num_epocs = 300

for i in range(0,len(cat_train_samples)):
	inputs.append(cat_train_samples[i])
	targets.append(0)
	inputs.append(deer_train_samples[i])
	targets.append(1)

training_size = int(0.8*len(targets))
validation_size = int(0.2*len(targets))


# for i in xrange(0,len(data)):
#     targets.append(data[i]['out'])
#     inputs.append(data[i]['inp'])
nn.train(inputs, targets, num_epocs)
c=0
for j in range(0,len(deer_test_samples)):
    
    s=nn.predict(deer_test_samples[j])
    if s == 1 :
    	c+=1
    s=nn.predict(cat_test_samples[j])
    if s == 0 :
    	c+=1
    # s1=data[j+training_size+validation_size]['out']
    # #print "Original: ",data[j+450]['out'], "Predicted: ", s, "Input: ",data[j+450]['inp']  
    # if s == s1:
    #     c+=1
#print j
print("Correctly identified test cases : ",c)
print("Accuracy : ",((c*1.0)/((j+1)*2))*100)

