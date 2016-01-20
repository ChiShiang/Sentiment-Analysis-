import pickle

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
		
	def svm_process_train(self, training_data, s_gamma = 0.10000000001, s_C = 1.0):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		svm_clf = svm.SVC(kernel = 'rbf', gamma = s_gamma, C = s_C).fit(training_data.X, training_data.Y)
		
		# 將訓練好的classifier model輸出 
		joblib.dump(svm_clf, './model/svm_clf.pkl')	
		setattr("svm_clf", svm_clf)

	def load_svm_clf(self, model):
		# 讀取外部以訓練好的分類器
		svm_clf = joblib.load(model_pkl)
		setattr('svm_clf', svm_clf)

	def nbc_process_train(self, training_data):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		nbc_clf = nbc.train(training_data)
		
		# 將訓練好的classifier model輸出 
		with open("./model/nbc_clf", 'wb') as nbc_clf_file:
			pickle.dump(nbc_clf, nbc_clf_file)

		setattr("nbc_clf", nbc_clf)
		
	def load_nbc_clf(self, model, data):
		# 讀取外部以訓練好的分類氣	
		with open(model, 'rb') as clf_file:
			nbc_clf = pickle.load(clf_file)
		
		setattr('nbc_clf', nbc_clf)