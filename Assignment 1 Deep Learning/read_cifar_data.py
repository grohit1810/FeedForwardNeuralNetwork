import numpy as np
import pickle

folder = "CIFAR_Dataset"
train_files = ["data_batch_"+str(i) for i in range(1,6) ]
# print files
test_file = "test_batch"

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def convert_rgb_to_grayscale_and_normalize(image_vector):
	grayscale_vector = []
	individual_spec_length = int(len(image_vector)/3) 
	for i in range(individual_spec_length):
		red_value = image_vector[i]
		green_value = image_vector[i + individual_spec_length]
		blue_value = image_vector[i+ (2*individual_spec_length)]
		# New grayscale image = ( (0.3 * R) + (0.59 * G) + (0.11 * B) ).
		grayscale_value = ((0.3*red_value) + (0.59*green_value) + (0.11*blue_value))
		grayscale_vector.append(grayscale_value)
	return NormalizeData(np.asarray(grayscale_vector))

# This function taken from the CIFAR website
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

deer_samples, cat_samples = {}, {}
deer_train_samples, cat_train_samples = [], []

for file in train_files :
	data = unpickle(folder+"/" + file)
	for i in range(0,len(data[b'labels'])):
		if data[b'labels'][i] == 3:
			cat_train_samples.append(convert_rgb_to_grayscale_and_normalize(data[b'data'][i].tolist()))
		if data[b'labels'][i] == 4:
			deer_train_samples.append(convert_rgb_to_grayscale_and_normalize(data[b'data'][i].tolist()))

# print(len(deer_train_samples), len(cat_train_samples))

# print(deer_train_samples[0],deer_train_samples[0].shape)
# print(cat_train_samples[0],cat_train_samples[0].shape)
deer_samples['train'] = deer_train_samples
cat_samples['train'] = cat_train_samples
# pickle.dump(pickle.dump(cat_train_samples, open("cat_train_samples.pkl", "wb")))
# pickle.dump(pickle.dump(deer_train_samples, open("deer_train_samples.pkl", "wb")))

cat_test_samples, deer_test_samples = [], []

test_data = unpickle(folder+"/" + test_file)
for i in range(0,len(test_data[b'labels'])):
	if test_data[b'labels'][i] == 3:
		cat_test_samples.append(convert_rgb_to_grayscale_and_normalize(test_data[b'data'][i].tolist()))
	if test_data[b'labels'][i] == 4:
		deer_test_samples.append(convert_rgb_to_grayscale_and_normalize(test_data[b'data'][i].tolist()))

deer_samples['test'] = deer_test_samples
cat_samples['test'] = cat_test_samples

pickle.dump(cat_samples, open("cat_samples.pkl", "wb"))
pickle.dump(deer_samples, open("deer_samples.pkl", "wb"))
