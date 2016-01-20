from feature_extraction import FeatureExtractModule as FEM
from preprocessing import * 
from training_testing_gen import * 

def feature_statistic(tweets):
	tdict = {}
	class_count_all = {'0':0, '1':0, '-1':0}
	for tweet in tweets:
		class_count_all[tweet.SA] += 1
		for word in tweet.Feature:
			if word not in tdict.keys():
				class_count = {'0':0, '1':0, '-1':0, "count": 1}
				class_count[tweet.SA] += 1
				tdict[word] = class_count
			else:
				tdict[word]['count'] += 1
				tdict[word][tweet.SA] += 1 
	with open("./test_analysis.txt", 'w') as tempfile:
		tempfile.write("{}:{}:{}:{}:{}\n".format('',len(tweets), class_count_all['0'], class_count_all['1'], class_count_all['-1']))
		for k, v in tdict.items():
			tempfile.write("{}:{}:{}:{}:{}\n".format(k, v['count'], v['0'], v['1'], v['-1']))


if __name__ == "__main__":
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

	# 製作訓練資料集與測試資料集 
	training_tweets, testing_tweets = training_testing_gen(tweets_box, probability_train = 0.7)

	# 從Tweet Feature中，統計出適用的字，作為特徵向量
	feature_statistic(training_tweets)

	print("Finish")











