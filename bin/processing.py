import time
import multiprocessing
from tweet import Tweet as t
from filereader import FileReader as fr

def preprocess(path, delmitier, col_tag, feature_extract_func):
	# 建立multiprocessing的pool
	# 若沒特別設定Pool(processes = ?) 則默默認為multiprocessing.cpu_count() e.g. 4 core...
	pool = multiprocessing.Pool()

	# 讀取tweet資料
	tweets = fr(path, delmitier, col_tag)
	
	# 將每一個tweet建立為一個結構化物件，利用pool來加速運作
	# 之後需要用.get()將apply_async的物件轉為我們所需要的Tweet
	tweets_box = [pool.apply_async(t, (tweet, feature_extract_func)).get() for tweet in tweets.data_contents]
	pool.close()
	pool.join()

	return tweets_box

def word_vector_base(tweets, filter_range):
	word_vector = []
	# 建立文字特徵向量

	tdict = {}
	# 建立字典統計字數
	for tweet in tweets:
		# 對所有tweet解析
		for word in tweet.Feature:
			# 若這個字已經存在 則對該字的字數統計量加一
			if word not in tdict.keys():
				tdict[word] = 1
			else:
				tdict[word] += 1 
	
	# 過濾出現次數小於filter_range的次數
	word_vector = [word_key for word_key, word_value in tdict.items() if word_value > filter_range]
	
	# 寫出檔案
	with open("./test_analysis.txt", 'w') as tempfile:
		for k in word_vector:
			tempfile.write("{}\n".format(k))
	return word_vector

def feature_statistic(tweets):
	# 進行字數統計用
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

def logfile(message):
	# 紀錄處理進程與事件問題
	
	# 輸出時間標記
	t = time.time()
	time_stamp = datetime.datetime.fromtimestamp(t).strftime('%Y-%m-%d %H:%M:%S')

	with open("./logfile.txt", 'a') as log_file:
		log_file.write("{}\t{}\n".format(time_stamp, message))
	print(message)

def feature_vector_create(clf_name, tweets, word_vector):
	# 建立multiprocessing的pool
	# 若沒特別設定Pool(processes = ?) 則默默認為multiprocessing.cpu_count() e.g. 4 core...
	pool = multiprocessing.Pool()

	# 進行特徵向量初始化
	vector_base, find_tag = feature_vercor_initial(clf_name, word_vector)

	# 特徵向量
	feature_vector = []

	# 每筆資料的正解
	standard_vector = []

	for tweet in tweets:
		temp_vector = pool.apply_async(vector_generate, (vector_base, find_tag, word_vector, tweet).get())
		feature_vector.append(temp_vector['vector'])
		standard_vector.append(temp_vector['SA'])

	return {'data': feature_vector, 'standard': standard_vector}

def feature_vercor_initial(clf_name, word_vector):
	# 取得word_vector的長度
	vector_len = len(word_vector)

	# 判斷要產生哪一種的分類器 特徵向量
	if clf_name == 'nbc':
		fvec = vector_initial_nbc(vector_len)
		find_tag = True
	elif clf_name == 'svm':
		fvec = vector_initial_svm(vector_len)
		find_tag = 1
	# 回傳特徵向量基底與替換符號
	return fvec, find_tag

def vector_initial_nbc(vlength):
	# 返回與word vector一樣長的Fasle特徵
	return [Fasle] * vlength

def vector_initial_svm(vlength):
	# 返回與word vector一樣長的0特徵
	return [0] * vlength

def vector_generate(temp_vector, find_tag, word_vector, tweet):
	tdict = {}
	temp = temp_vector
	for word in tweet.Feature:
		for k, v in enumerate(word_vector):
			if v == word:
				temp[k] = find_tag
	tdict['vector'] = temp
	tdict['SA'] = tweet.SA
	return tdict





