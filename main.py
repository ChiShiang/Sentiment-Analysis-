from bin.feature_extraction import FeatureExtractModule as FEM
from bin.processing import * 
from bin.training_testing_gen import * 
from bin.classifier import Classifier as clf

if __name__ == "__main__":
	try:
		# Tweet 位置
		tweets_path = "./src/tweets.txt"
		logfile(message = "Loading tweets file! ...")


		# 資料分割符號
		delmitier = "|"
		
		# 要用欄位資訊
		col_tag = ['Subject', 'Content', 'SA']
		logfile(message = "Setting the column title in tweets file! ...")

		# 建立特徵擷取模組以及指定擷取函式
		FE_module = FEM()

		# 建立分類器
		CLF_model = clf()

		# 你的檔案中務必將你的tweet csv file中的tweet欄位設定為Content
		# Tweets_box屬性會包含你所設定的欄位資訊
		# 因此當你有多個欄位要使用時，只需要更改col_tag裡面的資訊
		# e.g.: tweets_box[0].Content 	 --> "Hi,....."
		# 		tweets_box[0].Subject 	 --> "Entertainment"
		# 		tweets_box[0].SA		 --> 1
		# 		tweets_box[0].Feature 	 --> ["xxx",...]
		tweets_box = preprocess(tweets_path, delmitier, col_tag, FE_module.Unigram_FEF)
		logfile(message = "tweets preprocessing has been completed! ...")

		# 製作訓練資料集與測試資料集 
		training_tweets, testing_tweets = training_testing_gen(tweets_box, probability_train = 0.7)
		logfile(message = "training and testing set generated! ...")

		# 從Tweet Feature中，統計出適用的字，作為特徵向量基底
		feature_base = word_vector_base(training_tweets, filter_range = 5)
		logfile(message = "word vector base generated!")

		# 從training data產生特徵向量
		training_data_svm = feature_vector_create('svm', training_tweets, feature_base)
		training_data_nbc = feature_vector_create('nbc', training_tweets, feature_base)
		logfile(message = "SVM and NBC training data has been generated!")

		# testing data產生特徵向量
		testing_data_svm = feature_vector_create('svm', testing_tweets, feature_base)
		testing_data_nbc = feature_vector_create('nbc', testing_tweets, feature_base)
		logfile(message = "SVM and NBC testing data has been generated!")

		# 訓練分類器並獲取訓練後的model
		CLF_model.svm_process_train(training_data_svm)
		logfile(message = "SVM with RBF kernel training has been completed!")

		CLF_model.nbc_process_train(training_data_nbc)
		logfile(message = "NB Classifier training has been completed!")

		# 測試分類器insample (training data) 與 outsample (testing data)
		insample_result_svm = CLF_model.svm_clf.predict(training_data_svm['data'])
		logfile(message = "SVM Insample testing has been completed!")

		insample_result_nbc = CLF_model.nbc_clf.predict(training_data_nbc['data'])
		logfile(message = "NBC Insample testing has been completed!")

		outsample_result_svm = CLF_model.svm_clf.predict(testing_data_svm['data'])
		logfile(message = "SVM Outsample testing has been completed!")

		outsample_result_nbc = CLF_model.nbc_clf.predict(testing_data_nbc['data'])
		logfile(message = "NBC Outsample testing has been completed!")

		logfile(message = "Show the classifier result!")

		CLF_model.classifier_report("SVM_IN", training_data_svm['standard'], insample_result_svm)
		CLF_model.classifier_report("NBC_IN", training_data_nbc['standard'], insample_result_nbc)

		CLF_model.classifier_report("SVM_OUT", testing_data_svm['standard'], outsample_result_svm)
		CLF_model.classifier_report("NBC_OUT", testing_data_nbc['standard'], outsample_result_nbc)

		logfile(message = "All processing has been finish!")

	except Exception as e:
		logfile(message = e)










