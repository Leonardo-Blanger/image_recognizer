from __future__ import print_function
import numpy as np 
import tensorflow as tf 
from six.moves import cPickle as pickle
import math
import time
import os

pickle_file = "faces.pickle"

with open(pickle_file, "rb") as f:
	save = pickle.load(f)

	train_dataset = save["train_dataset"]
	train_labels = save["train_labels"]

	valid_dataset = save["valid_dataset"]
	valid_labels = save["valid_labels"]

	test_dataset = save["test_dataset"]
	test_labels = save["test_labels"]

	del save
	print("Training set", train_dataset.shape, train_labels.shape)
	print("Validation set", valid_dataset.shape, valid_labels.shape)
	print("Test set", test_dataset.shape, test_labels.shape)

image_size = train_dataset.shape[1]
num_labels = 28
num_chanels = 1

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, image_size, image_size, num_chanels)).astype(np.float32)
	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
	return dataset, labels

train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)

print("Training set", train_dataset.shape, train_labels.shape)
print("Validation set", valid_dataset.shape, valid_labels.shape)
print("Test set", test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
	return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]

batch_size = 32
patch_size = 10
depth = 16
num_hidden = 512
valid_size = 100

graph = tf.Graph()

with graph.as_default():

	# Input
	tf_inputs = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_chanels))
	tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

	tf_inputs_valid = tf.placeholder(tf.float32, shape=(valid_size, image_size, image_size, num_chanels))
	tf_labels_valid = tf.placeholder(tf.float32, shape=(valid_size, num_labels))

	tf_keep_prob_dropout = tf.placeholder(tf.float32)

	tf_valid_dataset1 = tf.constant(valid_dataset[:500, :, :, :])
	tf_valid_dataset2 = tf.constant(valid_dataset[500:1000, :, :, :])
	tf_valid_dataset3 = tf.constant(valid_dataset[1000:, :, :, :])

	tf_test_dataset1 = tf.constant(test_dataset[:500, :, :, :])
	tf_test_dataset2 = tf.constant(test_dataset[500:1000, :, :, :])
	tf_test_dataset3 = tf.constant(test_dataset[1000:, :, :, :])

	layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_chanels, depth], stddev=0.1))
	layer1_biases = tf.Variable(tf.zeros([depth]))

	layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	layer2_biases = tf.Variable(tf.zeros([depth]))

	layer3_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
	layer3_biases = tf.Variable(tf.zeros([depth]))

	layer4_weights = tf.Variable(tf.truncated_normal([(image_size//8 + 1) * (image_size//8 + 1) * depth, num_hidden], stddev=0.1))
	layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))

	layer5_weights = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
	layer5_biases = tf.Variable(tf.constant(1.0, shape=[num_labels]))

	def model(data):
		conv = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding="SAME")
		hidden = tf.nn.max_pool(tf.nn.relu(conv + layer1_biases), [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

		conv = tf.nn.conv2d(hidden, layer2_weights, [1, 1, 1, 1], padding="SAME")
		hidden = tf.nn.max_pool(tf.nn.relu(conv + layer2_biases), [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

		conv = tf.nn.conv2d(hidden, layer3_weights, [1, 1, 1, 1], padding="SAME")
		hidden = tf.nn.max_pool(tf.nn.relu(conv + layer3_biases), [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")

		shape = hidden.get_shape().as_list()
		reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
		hidden = tf.nn.relu(tf.matmul(reshape, layer4_weights) + layer4_biases)

		hidden = tf.nn.dropout(hidden, tf_keep_prob_dropout)		
		return tf.matmul(hidden, layer5_weights) + layer5_biases

	logits = model(tf_inputs)
	valid_logits = model(tf_inputs_valid)

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_labels)) + 0.001 * (
		tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer4_weights))

	valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(valid_logits, tf_labels_valid)) + 0.001 * (
		tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer4_weights))
	
	#optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

	global_step = tf.Variable(0, trainable=False)
	learning_rate = tf.train.exponential_decay(0.01, global_step, 500, 0.9, staircase=False)
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(valid_logits)

	valid_prediction1 = tf.nn.softmax(model(tf_valid_dataset1))
	valid_prediction2 = tf.nn.softmax(model(tf_valid_dataset2))
	valid_prediction3 = tf.nn.softmax(model(tf_valid_dataset3))

	test_prediction1 = tf.nn.softmax(model(tf_test_dataset1))
	test_prediction2 = tf.nn.softmax(model(tf_test_dataset2))
	test_prediction3 = tf.nn.softmax(model(tf_test_dataset3))

num_steps = 2000

training_cost = []
validation_cost = []

training_accuracy = []
validation_accuracy = []

test_accuracy = 0

best_validation_accuracy = -1
best_validation = 0

saver = tf.train.Saver([
	layer1_weights, layer1_biases,
	layer2_weights, layer2_biases,
	layer3_weights, layer3_biases,
	layer4_weights, layer4_biases,
	layer5_weights, layer5_biases
])

model_path = "/tmp/model.ckpt"
save_path = ""

with tf.Session(graph=graph) as session:
	tf.global_variables_initializer().run()
	print("Initialized")
	beginning = time.time()

	def evaluate_validation_set(_step):
		#correct = np.sum(np.argmax(valid_prediction1.eval(), 1) == np.argmax(valid_labels[:500], 1)) + np.sum(np.argmax(valid_prediction2.eval(), 1) == np.argmax(valid_labels[500:1000], 1)) + np.sum(np.argmax(valid_prediction3.eval(), 1) == np.argmax(valid_labels[1000:], 1))
		correct = 0
		total_loss = 0.0

		for step in range(valid_dataset.shape[0]/valid_size):
			offset = step * valid_size
			batch_data = valid_dataset[offset:(offset + valid_size), :, :, :]
			batch_labels = valid_labels[offset:(offset + valid_size), :]

			feed_dict = {
				tf_inputs_valid : batch_data,
				tf_labels_valid : batch_labels,
				tf_keep_prob_dropout : 1.0
			}

			l, predictions = session.run([valid_loss, valid_prediction], feed_dict=feed_dict)

			correct += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))
			total_loss += l

		print("--------------- Validation accuracy: %d / %d = %.3f%c" % 
			(correct, valid_dataset.shape[0], 100.0 * correct / valid_dataset.shape[0], '%'))

		validation_cost.append(total_loss / (valid_dataset.shape[0]/valid_size))
		validation_accuracy.append(correct)

		global best_validation_accuracy
		global best_validation
		global save_path
		global model_path

		if(correct > best_validation_accuracy):
			best_validation_accuracy = correct
			best_validation = _step
			save_path = saver.save(session, model_path)

	def evaluate_test_set():
		#correct = np.sum(np.argmax(test_prediction1.eval(), 1) == np.argmax(test_labels[:500], 1)) + np.sum(np.argmax(test_prediction2.eval(), 1) == np.argmax(test_labels[500:1000], 1)) + np.sum(np.argmax(test_prediction3.eval(), 1) == np.argmax(test_labels[1000:], 1))
		correct = 0
		total_loss = 0.0

		for step in range(test_dataset.shape[0]/valid_size):
			offset = step * valid_size
			batch_data = test_dataset[offset:(offset + valid_size), :, :, :]
			batch_labels = test_labels[offset:(offset + valid_size), :]

			feed_dict = {
				tf_inputs_valid : batch_data,
				tf_labels_valid : batch_labels,
				tf_keep_prob_dropout : 1.0
			}

			l, predictions = session.run([valid_loss, valid_prediction], feed_dict=feed_dict)

			correct += np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1))
			total_loss += l

		print("--------------- Final Test accuracy: %d / %d = %.3f%c" % 
			(correct, test_dataset.shape[0], 100.0 * correct / test_dataset.shape[0], '%'))

	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]

		print("[", offset, ",", offset+batch_size, "]")

		feed_dict = {
			tf_inputs : batch_data,
			tf_labels : batch_labels,
			tf_keep_prob_dropout : 0.3
		}

		_, l, predictions = session.run([optimizer, loss, prediction], feed_dict=feed_dict)
		
		print("Minibatch loss at step %d: %f" % (step, l))
		print(np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1)), "/", predictions.shape[0])

		training_cost.append(l)
		training_accuracy.append(np.sum(np.argmax(predictions, 1) == np.argmax(batch_labels, 1)))

		if(step % 100 == 0):
			evaluate_validation_set(step)

	print("Final validation accuracy:")
	evaluate_validation_set(num_steps)
	print("Best validation accuracy: %d on iteration %d" % (best_validation_accuracy, best_validation))
	
	print("Training phase took %f seconds" % (time.time() - beginning))


	print("Final test accuracy:")
	saver.restore(session, model_path)
	evaluate_test_set()

pickle_file = "evaluations.pickle"

try:
	f = open(pickle_file, "wb")
	save = {
		"training_cost" : training_cost,
		"validation_cost" : validation_cost,
		"training_accuracy" : training_accuracy,
		"validation_accuracy" : validation_accuracy,
		"test_accuracy" : test_accuracy
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()
except Exception as e:
	print("Unable to save data to", pickle_file, ":", e)
	raise

statinfo = os.stat(pickle_file)
print("Compressed pickle size:", statinfo.st_size)