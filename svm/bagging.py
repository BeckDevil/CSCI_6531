import pandas as pd
import numpy as np
from random import seed, random, randrange
from evaluate import *
from svm import *

def subsample(X, y, ratio = 1.0):
	X_train = []
	X_test = []
	y_train = []
	y_test = []

	selected_indices = []
	n_sample = round(len(X)*ratio)
	while len(X_train) < n_sample:
		index = randrange(len(X))
		X_train.append(X[index])
		y_train.append(y[index])
		selected_indices.append(index)
	for index in range(len(X)):
		if index not in selected_indices:
			X_test.append(X[index])
			y_test.append(y[index])
	X_train = np.array(X_train)
	X_itest = np.array(X_test)
	y_train = np.array(y_train)
	y_test = np.array(y_test)
	return X_train, X_test, y_train, y_test

def mean(numbers):
	return sum(numbers)/float(len(numbers))



def main():
	# data =load data
	datasets = []
	data = pd.read_csv('adult.csv')
	print(data.shape)
	data.count()[1]
#	print(data.head())
	def cc(x):
		return sum(x == '?')
#	print(data.apply(cc))

	df = data[data.occupation != '?']
	#print(df.shape)

	df = df[df.workclass != '?']
	#print(df.shape)

	df = df[df['native.country'] != '?']
	#print(df.shape)
    #print(df.groupby(by='education')['education.num'].mean())

	df.loc[df['native.country'] != 'United-States', 'native.country'] = 'non_usa'
	df.loc[df['income'] == '<=50K', 'income'] = -1
	df.loc[df['income'] == '>50K', 'income'] = 1

	features_categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
	features_numerical = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']

	# convert the categorical features into one-hot encoding
	for feature in features_categorical:
		df1 = pd.get_dummies(df[feature], drop_first = False)
		df = df.drop([feature], axis = 1)
		df = df.join(df1)
		print(df.shape)

	# normalize the numerical features by z- normalization
	for feature in features_numerical:
		df[feature] = (df[feature] - df[feature].mean())/df[feature].std()

	#df['capital.change'] = (df['capital.gain'] > 0) | (df['capital.loss'] >0)
	#df['capital.change'] = df['capital.change'].astype(int)


	print(df.columns)
	print(df.head())

	# first test on hours.per.week, education.num
	df1 = df.drop(['income'], axis = 1)
	allX = df1.values
	allX.astype(float)
	ally = df.as_matrix(columns = ['income'])

	print(allX.shape, ally.shape)
	X = allX[0:2000]
	y = ally[0:2000]
	myC = 10

	num_ensamble = 10
	classifiers = []
	for i in range(num_ensamble):
		classifier = svm(C = myC, kernel = linear_kernel, gamma = 0.05, coef = 1)
		classifiers.append(classifier)

	for i in range(num_ensamble):
		X_train, X_val, y_train, y_val = subsample(X, y, 1.0)
		lagr_mult = classifiers[i].fit(X_train, y_train)

		y_pred = classifiers[i].predict(X_val)

		accuracy = get_accuracy(y_val, y_pred)
		print("Out of bag Validation accuracy is {}".format(accuracy))

	# while testing, predict with each svm
	# Take majority vote
	# measure accuracy
	X_test = allX[2001:4000]
	y_test = ally[2001:4000]
	predictions = []
	for i in range(num_ensamble):
		y_pred = classifiers[i].predict(X_test)
		predictions.append(y_pred)

	# do majority vote reduction
	predictions = np.array(predictions)
	pred_t = []
	for i in range(len(X_test)):
		myarray = predictions[:, i].reshape(-1)
		# print(myarray)
		u, indices = np.unique(myarray, return_inverse = True)
		pred_t.append(u[np.argmax(np.bincount(indices))])

	# calculate the accuracy
	accuracy = get_accuracy(pred_t, y_test)
	print("Testing Accuracy is ", accuracy)

if __name__ == "__main__":
	'''
	dataset = [1, 2, 2, 3, 1, 4, 5, 6, 6, 7, 8, 9, 1, 2]
	y = 	  [0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0]
	for i in range(5):
		a, b, c, d = subsample(dataset, y, 1.0)
		print(a)
		print(b)
		print(c)
		print(d)
		#print(mean(mysample))
	'''
	main()
