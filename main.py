from feature_extraction import FeatureExtractModule as FEM
from processing import * 
from training_testing_gen import * 

if __name__ == "__main__":
	try:
		# Tweet 位置
		tweets_path = "./tweets.txt"
		
		# 資料分割符號
		delmitier = "|"
		
		# 要用欄位資訊
		col_tag = ['Subject', 'Content', 'SA']

		# 建立特徵擷取模組以及指定擷取函式
		FE_module = FEM()

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

		# 從Tweet Feature中，統計出適用的字，作為特徵向量
		feature_base = word_vector_base(training_tweets, filter_range = 5)
		logfile(message = "word vector base generated!")

		logfile(message = "Finish")
	except Exception, e:
		logfile(message = e)










