import pickle 

lstm_x_output
Y_validation 
X_test
Y_test

def load():
	file_Name = 'lstm_output'
	f = open(file_Name, "wb")
	lstm_x_output = pickle.load(f)
	Y_validation = pickle.load(f) 
	X_test = pickle.load(f)
	Y_test = pickle.load(f)
	f.close()

def text_process(): 
	"""
	fetch a dataset in CoNLL 2000 format
	"""	
		

def train():
	lstm_x_output 

def score():
	

if __name__ == '__main__':
	load() 	
	text_process()
	train()
	
