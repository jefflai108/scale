from sklearn import mixture
from mfcc_hdf5 import combine_mfcc_id
import pickle 

def load_data():
	train_speech, train_noise, train_noisy, test_speech, test_noise, test_noisy = combine_mfcc_id(2)
	f = open('baseline_data', 'wb')
	pickle.dump(train_speech, f)
	pickle.dump(train_noise, f)
	pickle.dump(train_noisy, f)
	pickle.dump(test_speech, f)
	pickle.dump(test_noise, f)
	pickle.dump(test_noisy, f)
	f.close()

def main():
	"""
	main gmm code 
	"""
	print('Loading data...')
	f = open('baseline_data','rb')
	train_speech = pickle.load(f)
	train_noise = pickle.load(f)
	train_noisy = pickle.load(f)
	test_speech = pickle.load(f)
	test_noise = pickle.load(f)
	test_noisy = pickle.load(f)
	f.close()

	print('Training model...')
	gmm_speech = mixture.GaussianMixture(n_components=128).fit(train_speech)
	gmm_noise = mixture.GaussianMixture(n_components=128).fit(train_noise)
	gmm_noisy = mixture.GaussianMixture(n_components=128).fit(train_noisy)

	print('Evaluate...')
	count, error = 0, 0
	for i in test_speech:
		count += 1
		#Compute the per-sample average log-likelihood of the given data X
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
		if biggest(score_s, score_n, score_c) != 'speech': 
			error += 1
	for i in test_noise:
		count += 1
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
		if biggest(score_s, score_n, score_c) != 'noise': 
			error += 1
	for i in test_noisy:
		count += 1
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
		if biggest(score_s, score_n, score_c) != 'noisy': 
			error += 1
	accuracy = (1-error*1.0/count)*100
	print("Classification accuracy for the GMMs is %.2f" % accuracy)
	return accuracy 
	
def biggest(a, b, c):
	max = a
	if b>max: 
		max = b
		if c>max: 
			max = c
	else:
		if c>max: 
			max = c
	if max == a:
		return 'speech'
	elif max == b:
		return 'noise'
	else: 
		return 'noisy' 

if __name__ == '__main__':
	load_data()
	main()	

