# filereader class 是一個讀取tweet csv的物件
# 可以將每一個文件轉換成一個有結構化的物件
import csv

class FileReader():
	"""docstring for FileReader"""
	def __init__(self, filepath, delimiter, data_col):
		super(FileReader, self).__init__()
		self.delimiter = delimiter
		self.data_col = data_col
		self.filepath = filepath
		self.data_contents = self.read_data()

	def read_data(self):
		temp_contents = []
		with open(self.filepath) as fileObj:
			# 使用DictReader將csv檔案的內容以Dict方式儲存
			file_contents = csv.DictReader(fileObj, delimiter=self.delimiter)
			for row in file_contents:
				temp_data = {}
				for col in self.data_col:
					# data_col儲存你將會用到的欄位的資訓
					temp_data[col] = row[col]
				temp_contents.append(temp_data)
		return temp_contents