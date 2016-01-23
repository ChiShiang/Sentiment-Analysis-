import pickle

# scikit learn
from sklearn import svm
from sklearn.naive_bayes import GaussianNB as gnbc
from sklearn.externals import joblib

# nltk package
from nltk.classify import NaiveBayesClassifier as nbc
import nltk.classify.util

class Classifier(object):
	"""docstring for Classifier"""
	def __init__(self):
		super(Classifier, self).__init__()
		SVM = svm.SVC(kernel = 'rbf')
		NBC = gnbc()
		setattr(svm, 'svm_clf', SVM)
		setattr(gnbc, 'nbc_clf', NBC)
	# ===========================SVM Classification===========================

	def svm_process_train(self, training_data, s_gamma = 0.10000000001, s_C = 1.0):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		self.svm_clf = svm.SVC(kernel = 'rbf', gamma = s_gamma, C = s_C).fit(training_data['data'],
		                                                                training_data['standard'])	
		# 將訓練好的classifier model輸出 
		joblib.dump(self.svm_clf, './model/svm_clf.pkl')

	def load_svm_clf(self, model):
		# 讀取外部以訓練好的分類器
		self.svm_clf = joblib.load(model)

	# ===========================Naive Bayes Classification===================
	def nbc_process_train(self, training_data):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		self.nbc_clf = gnbc().fit(training_data['data'], training_data['standard'])
		# 將訓練好的classifier model輸出
		joblib.dump(self.nbc_clf, './model/nbc_clf.pkl')

	def load_nbc_clf(self, model):
		# 讀取外部以訓練好的分類器
		self.nbc_clf = joblib.load(model)

