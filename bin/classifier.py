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
		
		return svm_clf

	def svm_trainModel_test(self, model, data):
		if type(model) is str:
			# 讀取外部以訓練好的分類氣
			svm_clf = joblib.load(model_pkl)
		else:
			# 讀取已經訓練好的分類器模型
			svm_clf = model

		# 載入testing data進行預測
		testing_result = svm_clf.predict(data)
		
		return testing_result

	def nbc_process_train(self, training_data):
		# training_data 為tweet元件，因此可以從Feature 與 SA取得你的特徵與正確答案
		nbc_clf = nbc.train(training_data)
		
		# 將訓練好的classifier model輸出 
		with open("./model/nbc_clf", 'wb') as nbc_clf_file:
			pickle.dump(nbc_clf, nbc_clf_file)

		return nbc_clf
		
	def nbc_trainModel_test(self, model, data):
		if type(model) is str:
			# 讀取外部以訓練好的分類氣
			with open(model, 'rb') as clf_file:
				nbc_clf = pickle.load(clf_file)
		else:
			# 讀取已經訓練好的分類器模型
			nbc_clf = model

		# 載入testing data進行預測
		testing_result = nbc_clf.prob_classify_many([feature for feature, sa in data])
		
		return testing_result 