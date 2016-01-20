import random as r

def training_testing_gen(tweets, probability_train):
	# 建立training set
	training_set = []

	# 建立testing set 
	testing_set = []

	for index in range(len(tweets)):
		# 利用random的seed來固定每一次random產生的值
		if training_seed(index, probability_train):
			training_set.append(tweets[index])
		else:
			testing_set.append(tweets[index])
	return training_set, testing_set

def training_seed(index, prob_train):
	# 用來固定產生亂數的規律性
	r.seed(index%10, 10)
	if r.random() < prob_train:
		return True
	else:
		return False