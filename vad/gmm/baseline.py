from sklearn import mixture
from mfcc_hdf5 import combine_mfcc_id
import pickle 
from sklearn.metrics import confusion_matrix

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
	output = open('gmm_128_log.txt', 'w')
	output.write('Log file for gmm with n_components=128\n')
	output.write('Loading data...\n')
	f = open('baseline_data','rb')
	train_speech = pickle.load(f)
	train_noise = pickle.load(f)
	train_noisy = pickle.load(f)
	test_speech = pickle.load(f)
	test_noise = pickle.load(f)
	test_noisy = pickle.load(f)
	f.close()

	output.write('Training model...\n')
	gmm_speech = mixture.GaussianMixture(n_components=128).fit(train_speech)
	gmm_noise = mixture.GaussianMixture(n_components=128).fit(train_noise)
	gmm_noisy = mixture.GaussianMixture(n_components=128).fit(train_noisy)

	print('Evaluate...')
	confusion_true, confusion_predicted = [], []
	for i in test_speech:
		confusion_true.append('speech')
		#Compute the per-sample average log-likelihood of the given data X
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
		log_likhod = biggest(score_s, score_n, score_c)
		if log_likhod == 'speech': 
			confusion_predicted.append('speech')
		elif log_likhod == 'noise':
			confusion_predicted.append('noise')
		else:
			confusion_predicted.append('noisy')
	for i in test_noise:
		confusion_true.append('noise')
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
                log_likhod = biggest(score_s, score_n, score_c)
                if log_likhod == 'speech':
                        confusion_predicted.append('speech')
                elif log_likhod == 'noise':
                        confusion_predicted.append('noise')
                else:
                        confusion_predicted.append('noisy')	
	for i in test_noisy:
		confusion_true.append('noisy')
		score_s = gmm_speech.score(i) 
		score_n = gmm_noise.score(i)
		score_c = gmm_noisy.score(i)	
                log_likhod = biggest(score_s, score_n, score_c)
                if log_likhod == 'speech':
                        confusion_predicted.append('speech')
                elif log_likhod == 'noise':
                        confusion_predicted.append('noise')
                else:
                        confusion_predicted.append('noisy')
	confusion = np.array(confusion_matrix(confusion_true, confusion_predicted, labels=['speech','noisy','noise']))
	output.write("Confusion_matrix is")
	output.write(confusion)
	accuracy = np.trace(confusion)/sum(sum(confusion))*100
	output.write("\nClassification accuracy for the GMMs is %.2f\n" % accuracy)
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
	#load_data()
	main()	

