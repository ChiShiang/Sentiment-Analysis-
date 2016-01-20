# scikit learn
from sklearn import svm
from sklearn.externals import joblib

# nltk package
from nltk.classify import NaiveBayesClassifier as nbc
import nltk.classify.util

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self):
		super(Classifier, self).__init__()
		
	def svm_process_train(self, training_data):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		

	def nbc_process_train(self, training_data):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		

	def svm_trainModel_test(self, model_file, testing_data):


	def nbc_trainModel_test(self, model_file, testing_data):