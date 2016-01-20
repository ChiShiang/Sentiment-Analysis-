from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
from nltk.util import ngrams

class FeatureExtractModule():
	# 你可以將你想要的特徵擷取函式寫在FeatureExtractModule
	# 因此，該模組只適合將建立FeatureExtraction_Func在這
	def __init__(self):
		self.stopword_set = stopwords.words('english')

	def Unigram_FEF(self, tweet):
		# 採用詞性
		pos_candidates = ['NN', 'JJ', 'VB', 'RB']

		# 將Tweet做sentence tokenize
		tweet_sents = sent_tokenize(tweet)

		# 進行word tokenize 以及標註詞性
		word_pos_set = [word_pos for sentence in tweet_sents for word_pos in pos_tag(word_tokenize(sentence))]

		# 將適合的字加入到feature
		feature = [word_pos[0].lower() for word_pos in word_pos_set
									   if word_pos[0].lower() not in self.stopword_set 
									   and word_pos[1][0:2] in pos_candidates]
		
		# 回傳特徵
		return feature

