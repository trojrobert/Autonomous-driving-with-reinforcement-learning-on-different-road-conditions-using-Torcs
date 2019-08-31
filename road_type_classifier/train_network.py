

# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import image
import argparse
import os

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, Conv1D



# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
	help="path to input dataset")
ap.add_argument("-m", "--model", required=True,
	help="path to output model")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output loss/accuracy plot")
#args = vars(ap.parse_args())

#cretate cnn to classifier the images
def cnn(dataset_classes):

	#assign dataset path
	base_dir = "/home/trojrobert/SRP/dataset"
	train_dir = os.path.join(base_dir, 'train')
	validation_dir = os.path.join(base_dir, 'validation')


	# Set import constant
	BATCH_SIZE = 20
	IMG_SIZE = 64
	EPOCHS = 20


	num_class = len(dataset_classes)

	train_image_generator = ImageDataGenerator(rescale=1. / 255)
	validation_image_generator = ImageDataGenerator(rescale=1. / 255)

	train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
															   directory=train_dir,
															   shuffle=True,
															   target_size=(IMG_SIZE, IMG_SIZE),  # (64, 64)
															   class_mode='sparse')
	val_data_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
																  directory=validation_dir,
																  shuffle=False,
																  target_size=(IMG_SIZE, IMG_SIZE),  # (64,64)
																  class_mode='sparse')

	# Create the model
	model = Sequential()
	conv1 = Conv2D(32, 3, 3, border_mode='same', activation = "relu", input_shape=(64, 64, 3))

	model.add(conv1)
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(64, 3, 3, border_mode='same', activation = "relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Conv2D(128, 3, 3, border_mode='same', activation = "relu"))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	#model.add(Dropout(0.2))
	model.add(Dense(512, activation='relu'))
	#model.add(Dropout(0.2))
	model.add(Dense(num_class, activation='softmax'))

	model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
	print(model.summary())

	history = model.fit_generator(train_data_gen, samples_per_epoch = 100,
	 					nb_epoch=EPOCHS,
	 					validation_data=val_data_gen, nb_val_samples= 50)

	acc = history.history['acc']
	val_acc = history.history['val_acc']

	loss = history.history['loss']
	val_loss = history.history['val_loss']

if __name__ == "__main__":
	dataset_classes =  ["asphalt", "dirt"]
	cnn(dataset_classes)

# # initialize the number of epochs to train for, initia learning rate,
# # and batch size
# EPOCHS = 25
# INIT_LR = 1e-3
# BS = 32
#
# # initialize the data and labels
# print("[INFO] loading images...")
# data = []
# labels = []
#
# # grab the image paths and randomly shuffle them
# imagePaths = sorted(list(paths.list_images(args["dataset"])))
# random.seed(42)
# random.shuffle(imagePaths)
#
# # loop over the input images
# for imagePath in imagePaths:
# 	# load the image, pre-process it, and store it in the data list
# 	image = cv2.imread(imagePath)
# 	image = cv2.resize(image, (28, 28))
# 	image = img_to_array(image)
# 	data.append(image)
#
# 	# extract the class label from the image path and update the
# 	# labels list
# 	label = imagePath.split(os.path.sep)[-2]
# 	label = 1 if label == "santa" else 0
# 	labels.append(label)
#
# # scale the raw pixel intensities to the range [0, 1]
# data = np.array(data, dtype="float") / 255.0
# labels = np.array(labels)
#
# # partition the data into training and testing splits using 75% of
# # the data for training and the remaining 25% for testing
# (trainX, testX, trainY, testY) = train_test_split(data,
# 	labels, test_size=0.25, random_state=42)
#
# # convert the labels from integers to vectors
# trainY = to_categorical(trainY, num_classes=2)
# testY = to_categorical(testY, num_classes=2)
#
# # construct the image generator for data augmentation
# aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
# 	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
# 	horizontal_flip=True, fill_mode="nearest")
#
# # initialize the model
# print("[INFO] compiling model...")
# model = LeNet.build(width=28, height=28, depth=3, classes=2)
# opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
# model.compile(loss="binary_crossentropy", optimizer=opt,
# 	metrics=["accuracy"])
#
# # train the network
# print("[INFO] training network...")
# H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
# 	validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
# 	epochs=EPOCHS, verbose=1)
#
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(args["model"])
#
# # plot the training loss and accuracy
# plt.style.use("ggplot")
# plt.figure()
# N = EPOCHS
# plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
# plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
# plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
# plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
# plt.title("Training Loss and Accuracy on Santa/Not Santa")
# plt.xlabel("Epoch #")
# plt.ylabel("Loss/Accuracy")
# plt.legend(loc="lower left")
# plt.savefig(args["plot"])