from nn_model import NN
import os
import shutil
import random
import numpy as np
import cPickle as pickle
from PIL import Image
from scipy import ndimage
from keras.models import Sequential
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import keras

class Recognizer(object):
	def __init__(self, width=100, height=100, channels=1,
		image_dir="images", dataset_dir="datasets", models_dir="models",
		clear_previous=False):

		self.width = width
		self.height = height
		self.channels = channels

		self.image_dir = image_dir
		self.dataset_dir = dataset_dir
		self.models_dir = models_dir

		if(clear_previous):
			self.clear()

		if(os.path.isdir(self.image_dir) == False):
			os.makedirs(self.image_dir)

		if(os.path.isdir(self.dataset_dir) == False):
			os.makedirs(self.dataset_dir)

		if(os.path.isdir(self.models_dir) == False):
			os.makedirs(self.models_dir)

		self.model = Sequential()
		self.model.add(keras.layers.InputLayer(input_shape=(self.width, self.height, self.channels)))

	def clear(self):
		if(os.path.isdir(self.image_dir)):
			shutil.rmtree(self.image_dir)

		if(os.path.isdir(self.dataset_dir)):
			shutil.rmtree(self.dataset_dir)

		if(os.path.isdir(self.models_dir)):
			shutil.rmtree(self.models_dir)

	def add_class(self, qt_images, pictures_path, name):
		print("Adding %s." % name)
		path = os.path.join(self.image_dir, "formated", name)

		if(os.path.isdir(path) == False):
			os.makedirs(path)

		qt = 0
		for picture in pictures_path:
			try:
				im = Image.open(picture)
			except:
				print("Error reading: %s Ignoring..." % picture)
				continue

			im = im.resize((self.width, self.height), Image.ANTIALIAS)
			im.save(os.path.join(path, "%s_%d.png"%(name, qt)))
			qt += 1

		while(qt_images and qt < qt_images):
			print("Generating more images: %d/%d" % (qt, qt_images))
			self.generate_more_images(qt_images, path, name)
			qt = len(os.listdir(path))
		print("Total: %d images." % qt)

		data = self.preprocess_data(path)
		pickle_file = os.path.join(self.dataset_dir, name)

		try:
			f = open(pickle_file, "wb")
			save = {
				"data" : data,
				"label" : name
			}
			pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
			f.close()
		except Exception as e:
			print("Unable to save data to %s: %s" % (pickle_file, e))
			raise

		statinfo = os.stat(pickle_file)
		print("Compressed pickle size: %s" % statinfo.st_size)

	def generate_more_images(self, qt_images, path, name):
		width = self.width
		height = self.height

		pictures = os.listdir(path)
		qt = len(pictures)

		for picture in pictures:
			im1 = Image.open(os.path.join(path, picture))
			pix1 = im1.load()

			for x in range(2,10,1):
				im2 = Image.open(os.path.join(path, picture))
				pix2 = im2.load()

				for i in range(width):
					for j in range(height):
						if(j < x): pix2[i,j] = 0
						else: pix2[i,j] = pix1[i,j-x]

				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break

				im2 = Image.open(os.path.join(path, picture))
				pix2 = im2.load()

				for i in range(width):
					for j in range(height):
						if(i < x): pix2[i,j] = 0
						else: pix2[i,j] = pix1[i-x,j]

				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break

				im2 = Image.open(os.path.join(path, picture))
				pix2 = im2.load()

				for i in range(width):
					for j in range(height):
						if(j+x >= height): pix2[i,j] = 0
						else: pix2[i,j] = pix1[i,j+x]

				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break

				im2 = Image.open(os.path.join(path, picture))
				pix2 = im2.load()

				for i in range(width):
					for j in range(height):
						if(i+x >= width): pix2[i,j] = 0
						else: pix2[i,j] = pix1[i+x,j]

				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break
			if(qt == qt_images): break

			for i in range(2,10,1):
				im2 = im1.rotate(-i)
				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break
				im2 = im1.rotate(i)
				im2.save(os.path.join(path, "%s_%d.png"%(name,qt)))
				qt += 1
				if(qt == qt_images): break
			if(qt == qt_images): break

	def preprocess_data(self, path):
		pictures = os.listdir(path)
		random.shuffle(pictures)

		dataset = np.ndarray(
			shape=(len(pictures), self.width, self.height, self.channels),
			dtype=np.float32
		)

		for (i, picture) in enumerate(pictures):
			picture = os.path.join(path, picture)

			try:
				image_data = (ndimage.imread(picture).astype(float) - 127.5) / 255.0
				dataset[i,:,:,:] = image_data.reshape((self.width, self.height, self.channels))
			except Exception as e:
				raise Exception("Error processing %s. Aborting. %s" % (picture, e))

		print("Full dataset tensor: ", dataset.shape)
		return dataset

	def fit_to_data(self, optimizer="rmsprop", batch_size=100, valid_split=0.33, valid_interval=10, num_steps=100, callbacks=[]):
		self.retrieve_data()
		self.model.add(keras.layers.Dense(self.num_labels, activation="softmax"))

		self.model.compile(
			optimizer = optimizer,
			loss = "categorical_crossentropy",
			metrics = ["accuracy"]
		)

		valid_size = int(len(self.dataset) * valid_split)

		valid_data = self.dataset[:valid_size]
		valid_labels = self.labels[:valid_size]

		train_data = self.dataset[valid_size:]
		train_labels = self.labels[valid_size:]

		loss, acc  = [], []
		val_loss, val_acc = [], []
		step = 0

		while(step < num_steps):
			for i in range(len(train_data) / batch_size):
				batch_data = train_data[i*batch_size : (i+1)*batch_size]
				batch_labels = train_labels[i*batch_size : (i+1)*batch_size]

				res = self.model.train_on_batch(x=batch_data, y=batch_labels)

				loss.append(res[0])
				acc.append(res[1])

				step += 1
				print("Step: %d\nLoss: %.4f\nAccuracy: %.2f\n\n" % (step, loss[-1], acc[-1]))

				if(step%valid_interval == 0):
					res = self.model.evaluate(x=valid_data, y=valid_labels)
					val_loss.append(res[0])
					val_acc.append(res[1])
					print("Validation: %d\nLoss: %.4f\nAccuracy: %.2f\n\n" % (step/valid_interval, val_loss[-1], val_acc[-1]))

				if(step == num_steps): break


		model_json = self.model.to_json()
		with open(os.path.join(self.models_dir, "model.json"), "w") as json_file:
		    json_file.write(model_json)
		self.model.save_weights(os.path.join(self.models_dir, "weights.h5"))

		return loss, acc, val_loss, val_acc

	def retrieve_data(self):
		self.num_labels = 0

		dataset = []
		dataset_size = 0

		for file in os.listdir(self.dataset_dir):
			print("Retrieving file %s." % file)

			with open(os.path.join(self.dataset_dir, file), "rb") as f:
				save = pickle.load(f)
				dataset.append(save["data"])
				dataset_size += len(dataset[-1])
				del save
			self.num_labels += 1

		if(self.num_labels == 0):
			print("No data available.")
			return

		self.dataset = np.ndarray(
			shape=(dataset_size, self.width, self.height, self.channels),
			dtype=np.float32
		)

		self.labels = np.zeros(
			shape=(dataset_size, self.num_labels),
			dtype=np.float32
		)

		pos = 0
		for label, data in enumerate(dataset):
			for x in data:
				self.dataset[pos] = x
				self.labels[pos][label] = 1.0
				pos += 1

		self.dataset, self.labels = self.randomize(self.dataset, self.labels)
		print("%d training samples of %d classes." % (dataset_size, self.num_labels))

	def randomize(self, dataset, labels):
		permutation = np.random.permutation(labels.shape[0])
		shuffled_dataset = dataset[permutation, :, :]
		shuffled_labels = labels[permutation, :]
		return shuffled_dataset, shuffled_labels

	def predict(self, paths):
		input = np.ndarray(
			shape = (len(paths), self.width, self.height, self.channels),
			dtype = np.float32
		)

		for i, path in enumerate(paths):
			try:
				im = Image.open(path)
			except:
				print("Error reading %s." % path)
				return
			im = im.resize((self.width, self.height), Image.ANTIALIAS)
			pix = im.load()

			for x in range(self.width):
				for y in range(self.height):
					if(self.channels == 1):
						input[i][x][y][0] = (pix[y,x] - 127.5)/255.0
					else:
						input[i][x][y][0], input[i][x][y][1], input[i][x][y][2] = (pix[y,x] - 127.5)/255.0

		json_file = open(os.path.join(self.models_dir, "model.json"), 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		self.model = model_from_json(loaded_model_json)
		self.model.load_weights(os.path.join(self.models_dir, "weights.h5"))

		output = self.model.predict(input)
		return output