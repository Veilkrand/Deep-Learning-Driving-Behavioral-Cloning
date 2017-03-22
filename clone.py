import csv
import cv2
import matplotlib
matplotlib.use('Agg') # For AWS compatibility
from matplotlib import pyplot as plt
import random
import numpy as np
from keras.preprocessing.image import random_shift, flip_axis
from sklearn.utils import shuffle

## DONE: Visualize normal distribution for steering angles
## TODO: Create a true generator for loading on the fly

## DONE: Load second set and add to measurements and images sequentially
## DONE: Discard samples with zero angle
## DONE: Flip

## DONE: Test Translation? how will it work with the cropping?

## DONE: Use all cameras
## DONE: Crop
## DONE: Data Exploration
## About augmentation: We can't do rotation easily. We could do random alpha artifacts (shadows), change luminosity to simulate day and night driving, translate the image.


# Read and load the CSV file
local_folder='sim data/'
local_csvfile='driving_log.csv'

# Subfolders where the additional data sets are
data_sets=[
	#'1/',  	# Full lap
	#'2/',	# Full lap backwards
	'3/',	# Red lanes
	'4/',	# Red lanes + Dirt road
	'5/', 	# Red lanes + Dirt road
	'6/',	# dificult curve
	'7/' 	# 2 Full laps better quality
] 


# Corrections to add left and right camera images
#left_camera_steer_correction=0.25
#left_camera_steer_correction=0.1  # It works pretty well
left_camera_steer_correction=0.25
right_camera_steer_correction=-0.25


#IMAGES_INPUT_SHAPE=(160,320,3)
#IMAGES_INPUT_SHAPE=(66,200,3)
IMAGES_INPUT_SHAPE=(128,128,3)


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
			#if float(line[3])<0.01:
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
# print('Samples Collection: {}'.format(len(measurerements)))



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

def show_histogram(x):
	# the histogram of the data
	#n, bins, patches = plt.hist(x, 50, normed=1, facecolor='green', alpha=0.75)
	n, bins, patches = plt.hist(x, 100, normed=1, facecolor='green', alpha=0.75)

	plt.xlabel('Distribution')
	plt.ylabel('Angles')
	plt.title('Steering angles')
	#plt.axis([40, 160, 0, 0.03])
	plt.grid(True)

	plt.show()


# Show a single image using CV2 and wait for 'q'
def show_image(image):
	cv2.imshow( "Display window", image)
	cv2.waitKey(0)

# ------


def process_sequential_batch_generator(X,y, batch_size=32,augmentation=False):

	N = len(y)
	batches_per_epoch = N // batch_size

	X,y=shuffle(X,y)

	i = 0

	while 1:
		start = i*batch_size
		end = start+batch_size - 1

		batch_X, batch_y = [], []

		for index in range(start,end):
			if (index>N-1): break
			measurement = y[index]
			image=X[index]
			if (augmentation):
				image, measurement = preprocess_augmentation(image,measurement)
			batch_X.append(image)
			batch_y.append(measurement)

		i += 1
		if (i == batches_per_epoch-1):
			# reset the index so that we can cycle over the data_frame again
			i = 0

		yield (np.array(batch_X), np.array(batch_y))

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

"""
_generator = process_sequential_batch_generator(images,measurerements,1000)
X_train,y_train=next(_generator)
show_histogram(y_train)
exit(0)
"""

# --------------------------




from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(images,measurerements, test_size=0.2, random_state=0)

assert len(X_val)==len(y_val)
print('Training datasets: {}'.format(len(y_train)))
print('Validation datasets: {}'.format(len(y_val)))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, ELU
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Cropping2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

def foo_model():
	model = Sequential()

	model.add(Flatten(input_shape=(64,64,3)))
	model.add(Dense(1))

	return model

def original_simple_model():
	global IMAGES_INPUT_SHAPE

	model = Sequential()

	# Cropping the images
	# model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))

	# images normalization and centered
	model.add(Lambda(
		lambda x: (x / 255.0) - 0.5, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	"""
	model.add(Cropping2D(
		cropping=((50,20), (0,0)), 
		input_shape=IMAGES_INPUT_SHAPE
	))
	"""

	# first set of CONV => RELU => POOL	
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	
	model.add(Flatten())

	model.add(Dense(500))
	model.add(Dense(120))
	model.add(Dense(84))

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

	"""
	model.add(Cropping2D(
		cropping=((50,20), (0,0)), 
		input_shape=IMAGES_INPUT_SHAPE
	))
	"""

	# first set of CONV => RELU => POOL	
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	# second set of CONV => RELU => POOL
	model.add(Convolution2D(6,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	
	model.add(Convolution2D(24,5,5,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Convolution2D(36,3,3,activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))


	model.add(Dropout(.5))
	
	model.add(Flatten())

	
	model.add(Dense(1024))
	model.add(Dropout(.5))
	model.add(Dense(500))
	model.add(Dense(120))
	model.add(Dense(84))
	model.add(Dense(32))

	model.add(Dense(1))

	return model

def nvidia_model2():
	
	"""
	Based on the Nvidia paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
	Modified with dropouts and several adjustments.
	Image size: (128x128x3)
	Input is normalized to around zero.

	The network has 24 layers plus output. There are 10 layers with learnable weights: 5 convolutional layers, and 5 fully connected layers.

	|	#	|	Type			|	Description													|
	|---	|---				|---															|
	|	1	|	Input			|	128x128x3 images input normalized to `zero`. 				|
	|	2	|	Convolution		|	24 5x5 convolutions with stride [2  2] and `valid` border. 	|
	|	3	|	Activation		|	ReLU activation. 											|
	|	4	|	Convolution		|	36 5x5 convolutions with stride [2  2] and `valid` border.  |
	|	5	|	Activation		|	ReLU activation. 											|
	|	6	|	Convolution		|	48 5x5 convolutions with stride [2  2] and `valid` border.  |
	|	7	|	Activation		|	ReLU activation. 											|
	|	8	|	Dropout			|	40% dropout chance. 										|
	|	9	|	Convolution		|	64 3x3 convolutions with stride [1  1] and `valid` border.  |
	|	10	|	Activation		|	ReLU activation. 											|
	|	11	|	Convolution		|	64 3x3 convolutions with stride [1  1] and `valid` border.  |
	|	12	|	Activation		|	ReLU activation. 											|
	|	13	|	Dropout			|	30% dropout chance. 										|
	|	14	|	Flatten			|	-															|
	|	15	|	Fully Connected	|	1024 fully connected layer. 								|
	|	16	|	Dropout			|	20% dropout chance. 										|
	|	17	|	Fully Connected	|	100 fully connected layer. 									|
	|	18	|	Activation		|	ReLU activation. 											|
	|	19	|	Fully Connected	|	50 fully connected layer. 									|
	|	20	|	Activation		|	ReLU activation. 											|
	|	21	|	Fully Connected	|	10 fully connected layer. 									|
	|	22	|	Activation		|	ReLU activation. 											|
	|	23	|	Fully Connected	|	1 fully connected layer. 									|
	|	24	|	Activation		|	Tanh activation. 											|
	|	--	|	Output			|	Steering angle. 											|
 
	"""




	model=Sequential()

	model.add(Lambda(
		lambda x: (x / 255.0) - 0.5, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Dropout(.4))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Dropout(.3))
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='tanh'))

	return model


def new_model():


	model = Sequential()

	model.add(Lambda(
		lambda x: (x / 255.0) - 0.5, 
		input_shape=IMAGES_INPUT_SHAPE
	))

	# layer 1 output shape is 32x32x32
	model.add(Convolution2D(32, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode="same"))
	model.add(ELU())

	# layer 2 output shape is 15x15x16
	model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
	model.add(ELU())
	model.add(Dropout(.5)) #4
	model.add(MaxPooling2D((2, 2), border_mode='valid'))

	# layer 3 output shape is 12x12x16
	model.add(Convolution2D(16, 3, 3, subsample=(1, 1), border_mode="valid"))
	model.add(ELU())
	#model.add(Dropout(.4)) #4

	# Flatten the output
	model.add(Flatten())

	# layer 4
	model.add(Dense(1024))
	model.add(Dropout(.3)) #4
	model.add(ELU())

	# layer 5
	model.add(Dense(512))
	model.add(ELU())

	# Finally a single output, since this is a regression problem
	model.add(Dense(1))

	model.compile(optimizer="adam", loss="mse")

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


def VGG16():

	model=Sequential()

	model.add(Lambda(
		lambda x: (x / 255.0) - 0.5, 
		input_shape=IMAGES_INPUT_SHAPE
	))




	model.add(Convolution2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Convolution2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
	model.add(Dropout(.4))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Convolution2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
	model.add(Dropout(.3))
	model.add(Flatten())
	model.add(Dense(1164, activation='relu'))
	model.add(Dropout(.2))
	model.add(Dense(100, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='tanh'))

	return model

# -----------------------

def main():
    # My code here
    pass

if __name__ == "__main__":
    main()


## ----------------------
# --- Training ---



"""
# Size 64x64

0.0057 with Nvidia2
batch_size=256
epochs=11
samples_per_epoch=(20000//batch_size)*batch_size
-
0.0056 with Nvidia2
batch_size=64
epochs=11
samples_per_epoch=(20000//batch_size)*batch_size
-
--------------
# Size 128x128

## Batch 64
Epoch 12/20
22050/22050 [==============================] - 31s - loss: 0.0048 - val_loss: 0.0046
Epoch 13/20
22050/22050 [==============================] - 30s - loss: 0.0045 - val_loss: 0.0052
Epoch 14/20
22050/22050 [==============================] - 30s - loss: 0.0045 - val_loss: 0.0044

Epoch 10/12
22050/22050 [==============================] - 30s - loss: 0.0054 - val_loss: 0.0050
Epoch 11/12
22050/22050 [==============================] - 30s - loss: 0.0051 - val_loss: 0.0048
Epoch 12/12
22050/22050 [==============================] - 30s - loss: 0.0051 - val_loss: 0.0052


## Batch 256
Epoch 12/12
22050/22050 [==============================] - 30s - loss: 0.0050 - val_loss: 0.0049

## Batch 64 (Winner??)
Epoch 12/12
22050/22050 [==============================] - 30s - loss: 0.0048 - val_loss: 0.0050

## Batch 512


"""

# Hyper parameters for manual tunning 

batch_size=64
#batch_size=256
#batch_size=512
#epochs=15
epochs=12
#samples_per_epoch=8192
#samples_per_epoch=4096
#samples_per_epoch=2048
samples_per_epoch=len(y_train)
#samples_per_epoch=(samples_per_epoch//batch_size)*batch_size
samples_per_epoch=22050



#_train_gen = process_batch_generator(X_train,y_train,batch_size,augmentation=True)
#_val_gen = process_batch_generator(X_val,y_val,batch_size,augmentation=False)

_train_gen = process_sequential_batch_generator(X_train,y_train,batch_size,augmentation=True)
_val_gen = process_sequential_batch_generator(X_val,y_val,batch_size,augmentation=False)


# -- Debugging generator results --
#result=next(_train_gen)
#print(result[0].shape[1:])


# -- Model Selection --
#model = foo_model()
#model = simple_model()
#model = nvidia_model()
model =nvidia_model2()
#model = new_model()

"""
# Optimizer forcing a learning rate
adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=adam)
"""

# Using Adam optimizer without forcing learning rate, Mean Square Error loss
model.compile(loss='mse', optimizer='adam')


# Save best losses helps manual model tunning
model_checkpoint = ModelCheckpoint(
        'model_best.h5',
        monitor='val_loss',
        verbose=0,
        save_best_only=True)

# Fit with my generators, for training and validation
history = model.fit_generator(
	_train_gen, 
	nb_epoch=epochs, 
	samples_per_epoch = samples_per_epoch, 
	validation_data= _val_gen,
	nb_val_samples= len(y_val),
	verbose = 1, 
	callbacks=[model_checkpoint]
	)

model.save('model.h5')

# Bell sound when training is over
print('\a\a\a\a\a\a\a')

# Summarize history for loss, and save chart image
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



