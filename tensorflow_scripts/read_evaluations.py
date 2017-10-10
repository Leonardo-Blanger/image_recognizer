from six.moves import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt

pickle_file = "evaluations.pickle"

with open(pickle_file, "rb") as f:
	save = pickle.load(f)

	training_cost = save["training_cost"]
	validation_cost = save["validation_cost"]

	training_accuracy = save["training_accuracy"]
	validation_accuracy = save["validation_accuracy"]

	test_accuracy = 1395

	del save

training_time = 4984.484802

plt.plot(training_cost, linewidth=2.0)
plt.xlabel('Iteration')
plt.ylabel('Error')
plt.xscale("log")
plt.show()