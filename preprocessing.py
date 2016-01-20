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