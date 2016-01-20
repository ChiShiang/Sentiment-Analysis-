# Tweet class 將每一則Tweet描述成一個物件
# 並且方便新增各種屬性擴充

class Tweet():
	"""docstring for Tweet"""
	# Content這個欄位，是Tweet class一定會有，也是資料欄位中必須要有的一個
	# 請檢查Tweet csv檔中，Tweet內容的欄位名稱是否為Content
	def __init__(self, tweet_box, feature_extract_func):
		super(Tweet, self).__init__()
		for key, value in tweet_box.items():
			setattr(self, key, value)
		self.Feature = feature_extract_func(self.Content)






		

		