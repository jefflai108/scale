import numpy as np 

def viterbi_alg(prior, transmat, obslik):
    #return the most-probable (Viterbi) path through 
    #the HMM state trellis
    T = obslik.shape[1]
    Q = prior.shape[0]
    delta, psi, path, scale = np.zeros((Q,T)), np.zeros((Q,T)), np.zeros((1,T)), np.zeros((1,T))
    
    t = 0
    #prior = np.transpose(prior)
    delta[:2,:2] = prior + obslik[:,t]
    psi[:,t] = 0
    for t in np.arange(1,T):
        for j in np.arange(0,Q):
            psi[j,t] = np.argmax(delta[:,t-1] + transmat[:,j])
            delta[j,t] = np.max(delta[:,t-1] + transmat[:,j])
            delta[j,t] += obslik[j,t]
    path[0,T-1] = np.argmax(delta[:,T-1])
    
    for t in np.arange(T-2,-1,-1):
        path[0,t] = psi[int(path[0,t+1]),t+1]
        #print(path[0,t])
    return np.transpose(path)

#if __name__ == '__main__':
