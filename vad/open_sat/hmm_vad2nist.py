import numpy as np 
from viterbi_path_mod import viterbi_alg
import random

def compute_hmm(ps_arr,pt_s,pt_ns,prior):
    """
    @ps: vector of probability of speech 
    """
    return_vad = []
    for p_s in ps_arr:
        p_ns=1-p_s #vector of probability of non_speech 
        p_s=p_s/prior 
        p_ns=p_ns/(1-prior)
        B = np.array([np.log(p_ns),np.log(p_s)])
        #print(B)
        prior_mat = np.array([np.log(1-prior),np.log(prior)])
        #print(prior_mat)
        transmat = np.array([[np.log(pt_ns),np.log(1-pt_ns)],[np.log(1-pt_s),np.log(pt_s)]])
        #print(transmat)
        vad = viterbi_alg(prior_mat, transmat, B)
        return_vad.append(vad)
    assert np.array(return_vad).shape == np.array(ps_arr).shape 
    return np.array(return_vad)

if __name__ == '__main__':
    ps = np.random.randn(100,100,1)  
    compute_hmm(ps,0.99,0.3,0.75)

