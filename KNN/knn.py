import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.decomposition import PCA

from collections import Counter
from multiprocessing import Pool, TimeoutError
import time
import os
import csv
from scipy import stats
from sklearn import preprocessing

class KNN:
	def __init__(self, dataset = None):
		self.dataset = dataset
		self.training_data = []
		self.testing_data = []

		self.training_labels = []
		self.testing_labels = []
		self.predictions = {}

	def load_data2(self):
		features = []
		data = []
		labels = []
		with open('diabetes.csv') as file_d:
			next(file_d)
			for line in file_d:
				seq = line.strip().split(',')
				seq = [float(x) for x in seq]
				labels.append(seq[-1])
				data.append(seq[0:-1])

		# Split the data into training and testing parts 0.75, 0.25 partition
		data = np.array(data)
		data_norm = data/data.max(axis = 0)
	#	data_norm = stats.zscore(data, axis = 0, ddof = 1)
		for i in range(len(data)):
			data_norm[i][1] = 3.00*data_norm[i][1]
		self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(data_norm, labels, test_size = 0.25, random_state = 123, stratify = labels)

		print(len(self.training_data))
		print(len(self.testing_data))

	def load_data(self):
		data = []
		labels = []
		with open('train.csv') as file_d:
			next(file_d)
			for line in file_d:
				seq = line.strip().split(',')
				seq = [int(x) for x in seq]
				#print(len(seq))
				labels.append(seq[0])
				data.append(seq[1:])
		print("Number of Training Samples: ", len(self.training_data))
	#	with open('test.csv') as file_d:
	#		next(file_d)
	#		for line in file_d:
	#			seq = line.strip().split(',')
	#			seq = [int(x) for x in seq]
	#			#print(len(seq))
	#			#self.testing_labels.append(seq[0])
	#			data.append(seq[0:])
	#	print("Number of Test instances: ", len(self.testing_data))

		transformer = PCA(n_components = 10)
		X_transformed = transformer.fit_transform(data)
		self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(X_transformed, labels, test_size = 0.20, random_state = 123, stratify = labels)
		print(len(self.training_data))
		print(len(self.testing_data))

	# TODO
	def show_tsne(self, number_to_show):
		plt_data = self.training_data[:number_to_show]
		plt_labels = self.training_labels[:number_to_show]

		for digit in range(10):
			instances = [i for i in plt_labels if i == digit]
			print("Digit {} appears {} times".format(digit, len(instances)))

		transformer = TSNE(n_components = 2, perplexity = 40, verbose = 2)
		fig, plot = plt.subplots()
		fig.set_size_inches(50, 50)
		plt.prism()

		X_transformed = transformer.fit_transform(plt_data)
		plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c= plt_labels)
		plt.tight_layout()
		count = 0
		for label, x, y in zip(plt_labels, X_transformed[:,0], X_transformed[:,1]):
			if count%100 == 0:
				plt.annotate(str(int(label)), xy = (x,y), color = 'black', weight = 'normal', size = 10, bbox = dict(boxstyle='round4, pad=.5'))
			count += 1
		plt.savefig('mnist_digits.pdf')

	def show_pca(self, number_to_show):
		plt_data = self.training_data[:number_to_show]
		plt_labels = self.training_labels[:number_to_show]

		for digit in range(10):
			instances = [i for i in plt_labels if i == digit]
			print("Digit {} appears {} times".format(digit, len(instances)))

		transformer = PCA(n_components = 2)
		fig, plot = plt.subplots()
		fig.set_size_inches(50, 50)
		plt.prism()

		X_transformed = transformer.fit_transform(plt_data)
		plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c= plt_labels)
		plt.tight_layout()
		count = 0
		for label, x, y in zip(plt_labels, X_transformed[:,0], X_transformed[:,1]):
			if count%100 == 0:
				plt.annotate(str(int(label)), xy = (x,y), color = 'black', weight = 'normal', size = 10, bbox = dict(boxstyle='round4, pad=.5'))
			count += 1
		plt.savefig('mnist_digits.pdf')

	def show_mds(self, number_to_show):
		plt_data = self.training_data[:number_to_show]
		plt_labels = self.training_labels[:number_to_show]

		for digit in range(10):
			instances = [i for i in plt_labels if i == digit]
			print("Digit {} appears {} times".format(digit, len(instances)))

		transformer = MDS(n_components = 2, max_iter = 100, n_init = 1)
		fig, plot = plt.subplots()
		fig.set_size_inches(50, 50)
		plt.prism()

		X_transformed = transformer.fit_transform(plt_data)
		plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c= plt_labels)
		plt.tight_layout()
		count = 0
		for label, x, y in zip(plt_labels, X_transformed[:,0], X_transformed[:,1]):
			if count%100 == 0:
				plt.annotate(str(int(label)), xy = (x,y), color = 'black', weight = 'normal', size = 10, bbox = dict(boxstyle='round4, pad=.5'))
			count += 1
		plt.savefig('mnist_digits.pdf')

	def distance(self, instance1, instance2):
		instance1 = np.array(instance1)
		instance2 = np.array(instance2)

		return np.linalg.norm(instance1 - instance2)

	def get_neighbors(self, test_instance, k):
		distances = []
		for index in range(len(self.training_data)):
			dist = self.distance(test_instance, self.training_data[index])
			distances.append((self.training_data[index], dist, self.training_labels[index]))
		distances.sort(key = lambda x: x[1])
		neighbors = distances[:k]
		return neighbors

	def vote(self, neighbors):
		class_counter = Counter()
		for neighbor in neighbors:
			class_counter[neighbor[2]] += 1

		return class_counter.most_common(1)[0][0]

	def vote_prob(self, neighbors):
		class_counter = Counter()
		for neighbor in neighbors:
			class_counter[neighbor[2]] += 1

		labels, votes = zip(*class_counter.most_common())
		winner = labels[0]
		votes4winner = votes[0]
		return winner, votes4winner/sum(votes)

	def vote_harmonic_weights(self, neighbors, all_results = True):
		class_counter = Counter()
		number_of_neighbors = len(neighbors)
		for index in range(number_of_neighbors):
			class_counter[neighbors[index][2]] += 1/(index+1)
		labels, votes = zip(*class_counter.most_common())

		winner = class_counter.most_common(1)[0][0]
		votes4winner = class_counter.most_common(1)[0][1]

		if all_results:
			total = sum(class_counter.values(), 0.0)
			for key in class_counter:
				class_counter[key] /= total
			return winner, class_counter.most_common()
		else:
			return winner, votes4winner/sum(votes)

	def vote_distance_weights(self, neighbors, all_results = True):
		class_counter = Counter()
		number_of_neighbors = len(neighbors)
		for index in range(number_of_neighbors):
			dist = neighbors[index][1]
			label = neighbors[index][2]
			class_counter[label] += 1/(dist**2 + 1)
		labels, votes = zip(*class_counter.most_common())

		# print(labels, votes)
		winner = class_counter.most_common(1)[0][0]
		votes4winner = class_counter.most_common(1)[0][1]

		if all_results:
			total = sum(class_counter.values(), 0.0)
			for key in class_counter:
				class_counter[key] /= total
			return winner, class_counter.most_common()
		else:
			return winner, votes4winner/sum(votes)


	def get_f1_score(self, predictions):
		TP = 0
		FP = 0
		TN = 0
		FN = 0
		for x in range(len(self.testing_labels)):
			if predictions[x] == 1:
				if self.testing_labels[x] == predictions[x]:
					TP += 1
				else:
					FP += 1
			elif predictions[x] == 0:
				if self.testing_labels[x] == predictions[x]:
					TN += 1
				else:
					FN += 1

		recall = TP/(TP+FN)
		precision = TP/(TP+FP)
		f1score = 2*TP / (2*TP+FP+FN)

		print(TP, FP, FN, TN)

		return recall, precision, f1score

	def get_confusion_matrix(self, predictions):
		confusion_matrix = {}
		for digit1 in range(10):
			confusion_matrix[digit1] = 0
		for x in range(len(predictions)):
			confusion_matrix[int(predictions[x])] += 1

		return confusion_matrix


	def get_accuracy(self, predictions):
		correct = 0
		for x in range(len(self.testing_labels)):
			if self.testing_labels[x] == predictions[x]:
				correct += 1
		return(correct/float(len(self.testing_labels)))


	def run_one(self, x):
		k = 5
		neighbors = self.get_neighbors(self.testing_data[x], k)
		label_p = self.vote(neighbors)
		self.predictions[x] = label_p


'''
if __name__ == "__main__":
	myKNN = KNN()
	myKNN.load_data2()

	predictions1 = []
	predictions2 = []
	predictions3 = []
	k = 7
	for x in range(len(myKNN.testing_data)):
		neighbors = myKNN.get_neighbors(myKNN.testing_data[x], k)
		label_p3, prob = myKNN.vote_distance_weights(neighbors)
		predictions3.append(label_p3)
		print("Predicted {}, Actual {}".format(label_p3, myKNN.testing_labels[x]))

	recall, precision, f1score = myKNN.get_f1_score(predictions3)
	print("Recall {}, Precision {}, and F1 Score {}".format(recall, precision, f1score))

'''
if __name__ == "__main__":
	pool = Pool(processes = 32)
	myKNN = KNN()
	myKNN.load_data()
#	myKNN.show_mds(5000)


	predictions1 = []
	predictions2 = []
	predictions3 = []
	k = 10
	#pool.map(myKNN.run_one, range(1000))

	print(myKNN.predictions)
	testdata = []
	for i, label in enumerate(myKNN.testing_labels):
		if label == 9:
			testdata.append(myKNN.testing_data[i])

	for x in range(len(testdata)):
		neighbors = myKNN.get_neighbors(testdata[x], k)
	#	label_p1, vote1 = myKNN.vote_prob(neighbors)
	#	label_p2, vote2 = myKNN.vote_harmonic_weights(neighbors)
		label_p3, vote3 = myKNN.vote_distance_weights(neighbors)
	#	predictions1.append(label_p1)
	#	predictions2.append(label_p2)
		predictions3.append(label_p3)
		print("*** Predicted label : {}, Actual label {}".format(label_p3, str(9)))
	confusion_matrix = myKNN.get_confusion_matrix(predictions3)
	print(confusion_matrix)

'''
#	accuracy = myKNN.get_accuracy(predictions)
	with open("submission1.csv", 'w') as res_file:
		csv_writer = csv.writer(res_file, delimiter = ',')
		csv_writer.writerow(["ImageId", "Label"])
		for i, pred in enumerate(predictions1):
			csv_writer.writerow([i, pred])
	with open("submission2.csv", 'w') as res_file:
		csv_writer = csv.writer(res_file, delimiter = ',')
		csv_writer.writerow(["ImageId", "Label"])
		for i, pred in enumerate(predictions2):
			csv_writer.writerow([i, pred])

	with open("submission3.csv", 'w') as res_file:
		csv_writer = csv.writer(res_file, delimiter = ',')
		csv_writer.writerow(["ImageId", "Label"])
		for i, pred in enumerate(predictions3):
			csv_writer.writerow([i, pred])
'''
