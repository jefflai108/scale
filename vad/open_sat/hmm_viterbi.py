
def transmat(prior, transmat, obslik):
	#return the most-probable (Viterbi) path through 
	#the HMM state trellis
	scaled = 0
	T = obslik.shape[1]
	Q = np.array(prior).shape[1]

	delta, psi, path, scale = np.zeros(Q,T), np.zeros(Q,T), np.zeros(1,T), np.zeros(1,T)

	t = 0
	delta = prior + obslik[:,t]
	psi[:,t] = 0
	for t in np.arange(1,T):
		for j in np.arange(0,Q):
			temp = max(delta(:,t-1) + transmat(:,j))
			delta[j,t] = temp[0]
			psi[j,t] = temp[1]
			delta[j,t] += obslik[j,t]
	temp = max(delta[:,T])
	p = temp[0]
	path[T-1] = temp[1]
	for t in np.arange(T-2,-1,-1):
		path[t] = psi(path(t+1),t+1)

