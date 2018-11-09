import os
import sys
import pandas as pd
import numpy as np
import cvxopt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from kernels import *
from evaluate import *

# Import sklearns for doint test train split and PCA
# SVM is built from scratch, not using any ML package
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA

class feature:
	def __init__(self, features):
		self.age = float(features[0])
		self.workclass = features[1].strip('"')
		self.fnlwgt = float(features[2])
		self.education = features[3]
		self.education_num = float(features[4])
		self.marital_status = features[5]
		self.occupation = features[6]
		self.relationship = features[7]
		self.race = features[8]
		self.sex = features[9]
		self.capital_gain = features[10]
		self.capital_loss = float(features[11])
		self.hours_per_week = float(features[12])
		self.native_country = features[13]
		self.income = features[14]


class svm:
	"""
	Parameters:
		C - regularization vs slack penalty
		kernel: a function, linear/polynomial or rbf
		power - degree of polynomial kernel
		gamma - parameter to rbf kernel function
		coef - bias term used in kernel function
	"""
	def __init__(self, C = 1, kernel = rbf_kernel, power = 6, gamma = 0.02, coef = 1):
		self.C = C
		self.kernel = kernel
		self.power = power
		self.gamma = gamma
		self.coef = coef

		self.lagr_multipliers = None
		self.support_vectors = None
		self.support_vector_labels = None
		self.intercept = None

	def fit(self, X, y):
		n_samples, n_features = np.shape(X)
		# print(n_samples, n_features)

		# set gamma to 1/#features by default
		if not self.gamma:
			self.gamma = 1/ n_features

		np.asarray(y)
		y = y.reshape(-1, 1) * 1
		X_dash = y * X
		H = np.dot(X_dash, X_dash.T) * 1

		# Initiate the kernel method with parameters
	#	self.kernel = self.kernel(power = self.power, gamma = self.gamma, coef = self.coef)

		# Calculate kernel matrix
	#	kernel_matrix = np.zeros((n_samples, n_samples))
	#	for i in range(n_samples):
	#		for j in range(n_samples):
	#			kernel_matrix[i, j] = self.kernel(X[i], X[j])
	#	print(kernel_matrix)

		# define the lagrange's quadratic optimization problem
	#	P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc = 'd')
		P = cvxopt.matrix(H)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1, n_samples) , tc = 'd')
		b = cvxopt.matrix(0, tc='d')

		if not self.C:
			G = cvxopt.matrix(np.identity(n_samples) * -1)
			h = cvxopt.matrix(np.zeros(n_samples))
		else:
			G_max = np.identity(n_samples) * -1
			G_min = np.identity(n_samples)
			G = cvxopt.matrix(np.vstack((G_max, G_min)))
			h_max = cvxopt.matrix(np.zeros(n_samples))
			h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
			h = cvxopt.matrix(np.vstack((h_max, h_min)))

		# solve the quadratic optimization problem using cvxopt
		minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers
		lagr_mult = np.array(minimization['x'])
	#	return lagr_mult


	#	print(lagr_mult)

		# Extract Support vectors
		# Get indices of non-zero lagrange multipliers
		idx = (lagr_mult > 1e-5).reshape(-1)
		# get the corresponding multipliers
		self.lagr_multipliers = lagr_mult[idx]
		# Get samples that will act as the support vectors
		self.support_vectors = X[idx]
		# Get the corresponding labels
		self.support_vector_labels = y[idx]
#		print(self.support_vectors)
#		print(self.support_vector_labels)

		# Calculate intercept with first support vector
		self.intercept = self.support_vector_labels[0]
		for i in range (len(self.lagr_multipliers)):
			self.intercept = self.intercept - self.lagr_multipliers[i] *\
					self.support_vector_labels[i] * \
					np.inner(self.support_vectors[i], self.support_vectors[0])
#		print(self.intercept)
		return lagr_mult

	def predict(self, X):
		y_pred = []
		#Iterate through list of samples and make predictions
		for sample in X:
			prediction = 0
			# Determine the label of the sample by using support vectors
			for i in range(len(self.lagr_multipliers)):
				prediction = prediction + self.lagr_multipliers[i] * self.support_vector_labels[i] * np.inner(self.support_vectors[i], sample)

			prediction += self.intercept
			y_pred.append(np.sign(prediction))
		return np.array(y_pred)

	def fitk(self, X, y):
		n_samples, n_features = np.shape(X)
		# print(n_samples, n_features)

		# set gamma to 1/#features by default
		if not self.gamma:
			self.gamma = 1/ n_features

		# Initiate the kernel method with parameters
		self.kernel = self.kernel(power = self.power, gamma = self.gamma, coef = self.coef)

		# Calculate kernel matrix
		kernel_matrix = np.zeros((n_samples, n_samples))
		for i in range(n_samples):
			for j in range(n_samples):
				kernel_matrix[i, j] = self.kernel(X[i], X[j])
	#	print(kernel_matrix)

		# define the lagrange's quadratic optimization problem
		P = cvxopt.matrix(np.outer(y, y) * kernel_matrix, tc = 'd')
		#P = cvxopt.matrix(H)
		q = cvxopt.matrix(np.ones(n_samples) * -1)
		A = cvxopt.matrix(y, (1, n_samples) , tc = 'd')
		b = cvxopt.matrix(0, tc='d')

		if not self.C:
			G = cvxopt.matrix(np.identity(n_samples) * -1)
			h = cvxopt.matrix(np.zeros(n_samples))
		else:
			G_max = np.identity(n_samples) * -1
			G_min = np.identity(n_samples)
			G = cvxopt.matrix(np.vstack((G_max, G_min)))
			h_max = cvxopt.matrix(np.zeros(n_samples))
			h_min = cvxopt.matrix(np.ones(n_samples) * self.C)
			h = cvxopt.matrix(np.vstack((h_max, h_min)))

		# solve the quadratic optimization problem using cvxopt
		minimization = cvxopt.solvers.qp(P, q, G, h, A, b)

		# Lagrange multipliers
		lagr_mult = np.array(minimization['x'])
	#	return lagr_mult


	#	print(lagr_mult)

		# Extract Support vectors
		# Get indices of non-zero lagrange multipliers
		idx = (lagr_mult > 1e-5).reshape(-1)
		# get the corresponding multipliers
		self.lagr_multipliers = lagr_mult[idx]
		# Get samples that will act as the support vectors
		self.support_vectors = X[idx]
		# Get the corresponding labels
		self.support_vector_labels = y[idx]
#		print(self.support_vectors)
#		print(self.support_vector_labels)

		# Calculate intercept with first support vector
		self.intercept = self.support_vector_labels[0]
		for i in range (len(self.lagr_multipliers)):
			self.intercept = self.intercept - self.lagr_multipliers[i] *\
					self.support_vector_labels[i] * \
					self.kernel(self.support_vectors[i], self.support_vectors[0])
#		print(self.intercept)
		return lagr_mult

	def predictk(self, X):
		y_pred = []
		#Iterate through list of samples and make predictions
		for sample in X:
			prediction = 0
			# Determine the label of the sample by using support vectors
			for i in range(len(self.lagr_multipliers)):
				prediction = prediction + self.lagr_multipliers[i] * self.support_vector_labels[i] * self.kernel(self.support_vectors[i], sample)

			prediction += self.intercept
			y_pred.append(np.sign(prediction))
		return np.array(y_pred)

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

	# COnvert sex into 0 and 1
	#df['sex'] = df['sex'].map({'Male': 0, 'Female': 1})
	#df['sex'] = df['sex'].astype(int)

	#df['marital.status'] = df['marital.status'].replace(['Never-married', 'Divorced', 'Separated', 'Widowed'], 'Single')
	#df['marital.status'] = df['marital.status'].replace(['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse'], 'Married')
	#df['marital.status'] = df['marital.status'].map({'Married': 1, 'Single' : 0})
	#df['marital.status'] = df['marital.status'].astype(int)

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
	X = []
	y = []
	lc = 0;
	df1 = df.drop(['income'], axis = 1)
	allX = df1.values
	allX.astype(float)
	ally = df.as_matrix(columns = ['income'])
	print(allX.shape, ally.shape)
	'''
	for index, row in df.iterrows():
		data = []
		data.append(row['hours.per.week'])
		data.append(row['education.num'])
		X.append(data)
		y.append(row['income'])
		if lc == 20:
			break
		else:
			lc = lc+1


	print(np.min(np.array(X)[:, 0]), np.max(np.array(X)[:, 1]))


	X = np.array(X)
	y = np.array(y)
	'''

#	fig, ax = plt.subplots()
#	ax.scatter(X[:, 0], X[:, 1], c = y[:])
	# mark the support vectors
	#ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c = support_vector_labels[:], s = 90)
#	plt.savefig('2-d.pdf')

	X = allX[0:4000]
	y = ally[0:4000]
	final_accuracy = []
	allC = [10]
	#X_train = X
	#y_train = y
	'''
	for myC in allC:
		classifier = svm(C = myC, kernel = linear_kernel, gamma = 0.05, coef = 1)
		lagr_mult = classifier.fit(X_train, y_train)
		fig, ax = plt.subplots()
		ax.scatter(X_train[:, 0], X_train[:, 1], c = y_train[:])

		# mark the support vectors
		#ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c = support_vector_labels[:], s = 90)

		#print(lagr_mult)
		w = np.sum(lagr_mult * y_train[:, None] * X_train, axis = 0)
		idx = (lagr_mult > 1*e-5).reshape(-1)
		b = y_train[idx] - np.dot(X_train[idx] , w)
		bias = b[0]
		support_vectors = X_train[idx]
		support_vector_labels = y_train[idx]
		ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c = support_vector_labels[:], s = 90)

		slope = -w[0] / w[1]
		#print(slope)
		intercept = -bias / w[1]
		#print(intercept)
		x = np.arange(-3.5,3.5, 0.5)
		ax.plot(x, x*slope + intercept, 'k-')
		filename = str(myC) + ".pdf"
		plt.savefig(filename)
	assert(0)
	'''
	'''
	skf = StratifiedKFold(n_splits = 10, random_state = 123, shuffle = True)
	fold = 0
	for myC in allC:
		accuracy = 0.0
		for train_index, val_index in skf.split(X, y):
			#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 123, stratify = y)
			X_train, X_val = X[train_index], X[val_index]
			y_train, y_val = y[train_index], y[val_index]

			classifier = svm(C = myC, kernel = linear_kernel, gamma = 0.05, coef = 1)
			lagr_mult = classifier.fit(X_train, y_train)

			# fig, ax = plt.subplots()
			# ax.scatter(X[:, 0], X[:, 1], c = y[:])
			# mark the support vectors
			# ax.scatter(support_vectors[:, 0], support_vectors[:, 1], c = support_vector_labels[:], s = 90)

			#print(lagr_mult)
			#w = np.sum(lagr_mult * y_train[:, None] * X_train, axis = 0)
			#idx = (lagr_mult > 1e-5).reshape(-1)
			#b = y_train[idx] - np.dot(X_train[idx] , w)
			#bias = b[0]
		#	# support_vectors = X_train[idx]
		#	# support_vector_labels = y_train[idx]

			#slope = -w[0] / w[1]
		#	#print(slope)
			#intercept = -bias / w[1]
			#print(intercept)
		#	x = np.arange(-3.5,3.5, 0.5)
		#	ax.plot(x, x*slope + intercept, 'k-')
		#	filename = str(fold) + "-fold.pdf"
		#	plt.savefig(filename)
		#	fold += 1

			y_pred = classifier.predict(X_val)
			accuracy += get_accuracy(y_val, y_pred)
			#print("Accuracy : ", accuracy)
		accuracy /= 10
		final_accuracy.append(accuracy)
		#print("Accuracy : ", accuracy)
	acc = np.asarray(final_accuracy, dtype = np.float32)
	print("Best Validation Accuracy is {} for C = {} :".format(np.max(acc), allC[np.argmax(acc)]))
	'''
	X_train = allX[0:1000]
	y_train = ally[0:1000]

	X_test = allX[1001:2000]
	y_test = ally[1001:2000]
	C = 10 # allC[np.argmax(acc)]

	classifier = svm(C = C, kernel = polynomial_kernel, gamma = 0.05, coef = 1)
	lagr_mult = classifier.fitk(X_train, y_train)

	y_pred = classifier.predictk(X_test)
	test_accuracy = get_accuracy(y_test, y_pred)
	recall, precision, f1score = f1_score(y_test, y_pred)

	print("accuracy: {}, recall: {}, precision{}, f1_score{}".format(test_accuracy, recall, precision, f1score))


'''
	fig, ((a, b, c, d), (e, f, g, h)) = plt.subplots(2, 4, figsize = (20,9))
	sns.countplot(df['workclass'], hue = df['income'], ax= h)
	sns.countplot(df['relationship'], hue = df['income'], ax= g)
	sns.countplot(df['marital.status'], hue = df['income'], ax= f)
	sns.countplot(df['race'], hue = df['income'], ax= e)
	sns.countplot(df['sex'], hue = df['income'], ax= d)
	sns.countplot(df['native.country'], hue = df['income'], ax= c)
	sns.countplot(df['education'], hue = df['income'], ax= b)
	sns.countplot(df['occupation'], hue = df['income'], ax= a)
	for ax in fig.axes:
		plt.sca(ax)
		plt.xticks(rotation=55)

	plt.subplots_adjust(hspace = 0.5, bottom = 0.2)
	plt.show("categorical.pdf")

	fig, ((a, b, c), (d, e, f)) = plt.subplots(2,3,figsize = (20, 8))
	sns.boxplot(y = 'hours.per.week', x='income', data = df, ax = a)
	sns.boxplot(y = 'age', x='income', data = df, ax = b)
	sns.boxplot(y = 'fnlwgt', x='income', data = df, ax = c)
	sns.boxplot(y = 'education.num', x='income', data = df, ax = d)
	sns.boxplot(y = 'capital.gain', x='income', data = df, ax = e)
	sns.boxplot(y = 'capital.loss', x='income', data = df, ax = f)
	plt.savefig("numerical.pdf")
'''


'''
	with open('adult.csv', 'r') as f:
		for line in f:
			seq = line.strip().split(',')
			for info in seq:


			print(seq)

#			data = []
#			#data = [float(seq[i]) for i in range(len(seq)-1)]
#			#data.append(float(seq[3])/float(seq[2]))
#			data.append(float(seq[-1]))
#			#data[-1] = 0.0 if data[-1] <= 5.0 else 1.0
#			data[-1] -= 1
#			datasets.append(data)
#	# normalize
#	# Replace and do test-train split


	X_train, X_test, Y_train, Y_test = train_test_split(data_norm, labels, test_size = 0.25, random_state = 123, stratify = labels)
	classifier = svm(kernel = kernels.polynomial_kernel, power = 4, coef = 1)
	classifier.fit(X_train, y_train)
	y_pred = classifier.predict(X_test)
	accuracy = evaluate.accuracy(y_test, y_pred)

	print("Accuracy : ", accuracy)

	# use PCA to reduce X_test to 2-dim and plot
i'''

if __name__ == "__main__":
	main()







