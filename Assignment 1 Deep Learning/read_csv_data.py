import csv
import pickle
import numpy as np
file = open("circles500.csv")
index = 0
data = []
for line in file :
	if index == 0:
		index +=1
		continue
	
	x0,x1,output_class = line.split(',')
	#print x0,x1,output_class
	current_row = {}
	inp = np.asarray([float(x0),float(x1)])
	out = np.zeros((2,))
	out[int(output_class)] = 1
	current_row["inp"] = inp
	current_row["out"] = int(output_class)
	data.append(current_row)

print(data)
pickle.dump(data, open("data.pkl", "wb"))
