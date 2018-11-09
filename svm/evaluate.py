import os

def f1_score(labels, predictions):
	TP = 0
	FP = 0
	TN = 0
	FN = 0
	for x in range(len(labels)):
		if predictions[x] == 1:
			if labels[x] == predictions[x]:
				TP += 1
			else:
				FP += 1
		elif predictions[x] == 0:
			if labels[x] == predictions[x]:
				TN += 1
			else:
				FN += 1

	recall = TP/(TP+FN)
	precision = TP/(TP+FP)
	f1score = 2*TP / (2*TP+FP+FN)

	print(TP, FP, FN, TN)

	return recall, precision, f1score

def get_accuracy(labels, predictions):
	correct = 0
	for x in range(len(labels)):
		if labels[x] == predictions[x]:
			correct += 1
	return(correct/float(len(labels)))
