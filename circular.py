class CircularList:
	def __init__(self, size):
		self.size = size
		self.items_list = []
		for i in range(size):
			self.items_list.append([])
		self.index = 0
	def append(self, x):
		self.items_list[self.index].append(x)
		self.index = (self.index+1)%self.size
	def clear(self):
		self.items_list = []
		for i in range(self.size):
			self.items_list.append([])
		self.index = 0
	def __str__(self):
		return str(self.items_list)
	def __getitem__(self, i):
		return self.items_list[i]
