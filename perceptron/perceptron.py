import numpy as np
import sys
from random import seed
from random import random
import csv

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

seed = 21
def predict(row, weights, isprint = False):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i+1]*row[i]
	prediction = 1.0 if activation > 0.5 else 0.0
	if isprint:
		print (activation, prediction)

	return prediction

def train_weights(datasets, train, learning_rate, n_epoch):
	# make the weights randon, 0.0 might sometimes get stuck in local minima
	weights = [float(0.0) for i in range(len(datasets[0]))]
	for epoch in range(n_epoch):
		# we are adjusting error at the end of an epoch, not stochastic
		sum_error = 0.0
		correct = 0
		#In each epoch, go over all training examples
		for row_id in train:
			prediction = predict(datasets[row_id], weights)
			if prediction == datasets[row_id][-1]:
				correct += 1
			# The error for individual training example
			error = datasets[row_id][-1] - prediction

			sum_error += (error ** 2)/len(train)

			# Adjust your weight, it's an stochastic in fact
			weights[0] = weights[0] + learning_rate * error
			for i in range(len(weights)-1):
				# W' <== W + alpha * L * x[i]
				weights[i+1] = weights[i+1] + learning_rate * error * datasets[row_id][i]

		accuracy = correct/len(train)

		print(">> Epoch {}, learning rate = {}, Training Error = {}".format(epoch, learning_rate, sum_error))
	return accuracy, weights

if __name__ == "__main__":
	datasets = []
	#labels = []
	with open('bupa_dataset', 'r') as f:
		for line in f:
			seq = line.strip().split(',')
			data = []
			#data = [float(seq[i]) for i in range(len(seq)-1)]
			#data.append(float(seq[3])/float(seq[2]))
			data.append(float(seq[-1]))
			#data[-1] = 0.0 if data[-1] <= 5.0 else 1.0
			data[-1] -= 1
			datasets.append(data)

	kfold = KFold(10, True, 1)
	for train, test in kfold.split(datasets):
		train_accuracy, weights = train_weights(datasets, train, 0.01, 500)
		correct = 0
		square_loss = 0.0
		for test_id in test:
			predicted = predict(datasets[test_id], weights, False)
			if predicted == datasets[test_id][-1]:
				correct += 1
			square_loss += ((predicted - datasets[test_id][-1])**2) / len(test)

		accuracy = correct / len(test)
		print("{} {}".format(train_accuracy, accuracy))

