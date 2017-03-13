import csv
import cv2
from matplotlib import pyplot as plt
import random
import numpy as np
from keras.preprocessing.image import random_shift, flip_axis
from sklearn.utils import shuffle

## DONE: Load second set and add to measurements and images sequentially
## DONE: Discard samples with zero angle
## DONE: Flip
## TODO: Test Translation? how will it work with the cropping?
## DONE: Use all cameras
## DONE: Crop
## DONE: Data Exploration
## About augmentation: We can't do rotation easily. We could do random alpha artifacts (shadows), change luminosity to simulate day and night driving, translate the image.



## Nvidia CNN 
### 256 batch
### with 3 Epochs? 10 Epochs?
### 20,000 images per Epoch

## Include Second lap of data?
## Visualize Loss on graph


"""
# Lesson learned
Validation doesn't need to be augmented. Batches need to be properly setup

"""

# Read and load the CSV file
local_folder='sim data/'
local_csvfile='driving_log.csv'

# Subfolders where the additional data sets are
data_sets=['1/','2/','3/','4/','5/'] 
#data_sets=['1/','2/'] 

# Corrections to add left and right camera images
#left_camera_steer_correction=0.25
left_camera_steer_correction=0.025
right_camera_steer_correction=-0.025

global IMAGES_INPUT_SHAPE
#IMAGES_INPUT_SHAPE=(160,320,3)
IMAGES_INPUT_SHAPE=(66,200,3)
#IMAGES_CV2_RESIZE=(IMAGES_INPUT_SHAPE[0],IMAGES_INPUT_SHAPE[1])



images=[]
measurerements=[]

print('Loading datasets...')
for data in data_sets:
	#Load CSV
	with open(local_folder+data+local_csvfile) as csvfile:
		reader = csv.reader(csvfile)
		for line in reader:

			# Skipping low steering values
			if float(line[3])==0.0:
			#if float(line[3])<0.1:
				continue

			# Prepare data and local paths
			for i in range(3): #load 0:center 1:left 2:right

				path=line[i]
				filename=path.split('/')[-1]
				local_path=local_folder+data+'IMG/'+filename
				image=cv2.imread(local_path)
				
				# Resize all images including validation sets
				image = cv2.resize(image, (IMAGES_INPUT_SHAPE[1],IMAGES_INPUT_SHAPE[0]))


				# Camera steering correction
				measurerement=float(line[3])
				if (i==1):
					measurerement+=left_camera_steer_correction
				elif(i==2):
					measurerement+=right_camera_steer_correction


				measurerements.append(measurerement)
				images.append(image)

				# show_image(image)
				# exit(0)



assert len(images)==len(measurerements)
print('Samples Collection: {}'.format(len(measurerements)))



# ------
# Helper functions for preprocessing and augmentation
def crop_image(image):
	h=int(image.shape[0])
	w=int(image.shape[1]) 
	#Crop [Y1:Y2, X1:X2] #(0,50)-(w,h-20)
	return image[50:h-20, 0:w] # Top 50px # Bottom 20px

def apply_image_random_brightness(image):
    image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    image = np.array(image, dtype = np.float64)
    bright = .5+np.random.uniform()
    image[:,:,2] = image[:,:,2]*bright
    image[:,:,2][image[:,:,2]>255]  = 255
    image = np.array(image, dtype = np.uint8)
    image = cv2.cvtColor(image,cv2.COLOR_HSV2RGB)
    return image

def apply_image_random_shadow(image):
	
	alpha=0.3

	h=image.shape[0]
	w=image.shape[1]

	points=np.array([ [0,0],[w-random.randint(0, w),0],[w-random.randint(0, w),h],[0,h] ], np.int32)

	overlay = image.copy()
	output=image.copy()
	
	#overlay=cv2.rectangle(image, (25, 25), (w-10, h-10), (0,0,0), -1)
	overlay=cv2.fillConvexPoly(image, points, (0,0,0))

	cv2.addWeighted(overlay, alpha, output, 1.0 - alpha,0, image)

	return image



# Sorted operations to improve performance
def preprocess_augmentation(image, measurement):

	global IMAGES_CV2_RESIZE

	# Crop Image
	#image = crop_image(image)

	# Image Vertical Shift by 20%
	# image_shifted=random_shift(image, 0, 0.2, 0, 1, 2)
	# images.append(image_shifted)
	# measurerements.append(measurerement)

	#global IMAGES_INPUT_SHAPE
	#image = cv2.resize(image, (IMAGES_INPUT_SHAPE[1],IMAGES_INPUT_SHAPE[0]))

	# Random brightness to simulate different light conditions
	image=apply_image_random_brightness(image)


	# Random shadow artefacts to improve generalization
	image=apply_image_random_shadow(image)


	# Flip the image 50% of the time
	if (random.randint(0, 100)>50):
		# Horizontal flip and steering reverse
		image = flip_axis(image, 1)
		measurement=-measurement


	return image,measurement
# -----
# Visualization helper functions
# Helper fuction to build a gallery from a image collection
def show_collection_gallery(collection,number=120):
	fig= plt.figure(figsize=(12,7))
	for i in range(number):   
	    fig.add_subplot(12,10,1+i)
	    image=random.choice(collection).squeeze()
	    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	    plt.imshow(image)
	    plt.axis("off")
		#fig.suptitle('Random preprocessed', fontsize=18)
	plt.show()

# Show a single image using CV2 and wait for 'q'
def show_image(image):
	cv2.imshow( "Display window", image)
	cv2.waitKey(0)

# ------



def process_batch_generator(X, y,batch_size=64,augmentation=False):

	X,y=shuffle(X,y)

	while 1:

		batch_X, batch_y = [], []

		for i in range(batch_size):
			index = random.randint(0, len(X) - 1)
			measurement = y[index]
			image=X[index]
			if (augmentation):
				image, measurement = preprocess_augmentation(image,measurement)
			batch_X.append(image)
			batch_y.append(measurement)

		batch_X,batch_y=shuffle(batch_X,batch_y)
		yield (np.array(batch_X), np.array(batch_y))


## Visualization
#_generator = process_batch_generator(images,measurerements,120)
#X_train,y_train=next(_generator)
#show_collection_gallery(X_train)
#exit(0)
	
'''
# Gallery of image results
fig = plt.figure()
for i in range(50):
	fig.add_subplot(10,5,1+i); plt.imshow(random.choice(images)); plt.axis("off");

plt.show()

exit(0)
'''


# --------------------------




from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images,measurerements, test_size=0.2, random_state=0)

assert len(X_val)==len(y_val)
print('Training datasets: {}'.format(len(y_val)))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam

def foo_model():
	model = Sequential()

	model.add(Flatten(input_shape=(64,64,3)))
	model.add(Dense(1))

	return model


def simple_model():
	global IMAGES_INPUT_SHAPE

	model = Sequential()

	# Cropping the images
	# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

	# images normalization and centered
	model.add(Lambda(
		lambda x: (x / 255.0) - 0.5, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	model.add(Cropping2D(
		cropping=((50,20), (0,0)), 
		input_shape=IMAGES_INPUT_SHAPE
	))

	# first set of CONV => RELU => POOL	
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(6,5,5,activation='relu'))
	

	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	# set of FC => RELU layers
	model.add(Flatten())

	model.add(Dense(500))

	# From video
	model.add(Dense(120))
	model.add(Dense(84))

	model.add(Dense(1))

	return model

def nvidia_model2():
	
	# They use 128x128 as image input

	# https://github.com/0bserver07/Nvidia-Autopilot-Keras

	# nb_epoch = 25
	# batch_size = 64

	global IMAGES_INPUT_SHAPE

	model=Sequential()

	model.add(Lambda(
		lambda x: x/127.5-1.0, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='tanh'))

	return model

def nvidia_model():
	global IMAGES_INPUT_SHAPE

	model=Sequential()


	model.add(Lambda(
		lambda x: x/127.5-1.0, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	#model.add(Cropping2D(
	#	cropping=((50,20), (0,0)), 
	#	input_shape=IMAGES_INPUT_SHAPE
	#))
	
	
	# #1 Convolutional layers with ELU activation
	model.add(Convolution2D(
			24, 5, 5, 
			subsample=(2,2), 
			border_mode="valid",
			init="he_normal"
	))	
	model.add(ELU())
	# #2 Convolutional layers with ELU activation
	model.add(Convolution2D(
			36, 5, 5, 
			subsample=(2,2), 
			border_mode="valid",
			init="he_normal"
	))	
	model.add(ELU())
	# #3 Convolutional layers with ELU activation
	model.add(Convolution2D(
			48, 5, 5, 
			subsample=(2,2), 
			border_mode="valid",
			init="he_normal"
	))	
	model.add(ELU())
	# #4 Convolutional layers with ELU activation
	model.add(Convolution2D(
			64, 3, 3, 
			subsample=(1,1), 
			border_mode="valid",
			init="he_normal"
	))	
	model.add(ELU())
	model.add(ELU())
	# #5 Convolutional layers with ELU activation
	model.add(Convolution2D(
			64, 3, 3, 
			subsample=(1,1), 
			border_mode="valid",
			init="he_normal"
	))	
	model.add(ELU())

	model.add(Flatten())

	# x4 fully-connected layers with ELU activation
	for i in [1164,100,50,10]:
		model.add(Dense(i,init="he_normal"))
		model.add(ELU())

	model.add(Dense(1,init="he_normal"))
	
	return model

## ----------------------
# Training

# Hyper parameters

#images_shape=(64,64,3) #?

#batch_size=64
batch_size=256
epochs=10
#epochs=3
#samples_per_epoch=8192
#samples_per_epoch=4096
#samples_per_epoch=2048
samples_per_epoch=20480



_train_gen = process_batch_generator(X_train,y_train,batch_size,augmentation=True)
_val_gen = process_batch_generator(X_val,y_val,batch_size,augmentation=False)

#result=next(_train_gen)
#print(result[0].shape[1:])


# Model Selection
#model = foo_model()

#model = simple_model()

model = nvidia_model()

adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
#model.fit(X_train, y_train, nb_epoch=epochs, validation_split=0.2,shuffle=True)

history = model.fit_generator(
	_train_gen, 
	nb_epoch=epochs, 
	samples_per_epoch = samples_per_epoch, 
	validation_data= _val_gen,
	nb_val_samples= len(y_val),
	verbose = 1, 
	)

model.save('model.h5')

# Bell done
print('\a\a\a\a\a\a\a')

# summarize history for loss
import time
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
time=str(int(time.time()))
plt.savefig(time+'-loss.png')
plt.show()



