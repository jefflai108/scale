import numpy as np 
import os 

"""
Author: Jeff Lai 
HLTCOE 2017 

Loop over every file in /export/b13/jlai/scale/vad/open_sat/ground_truth/dev_ref , retrieve n vad array for each file, create a dictionary to store all arrays 
"""

def vad_import():
	#Loop over the directory 
	dic = {}
	directory = "/export/b13/jlai/scale/vad/open_sat/ground_truth/ground_truth"
	vad_key = []
	for filename in os.listdir(directory):
		key = filename.split('.')[0] #filename as dictionary key 
		vad_key.append(key)
		with open("/export/b13/jlai/scale/vad/open_sat/ground_truth/ground_truth/" + filename) as f: 
			content = f.readlines()
		content = [x.strip() for x in content]
		
		temp = [] #vad array for a file 
		diff, accumulator=0, 0 
		for i in content: 
			start = float(i.split('\t')[2])*100 #in 10ms
			end = float(i.split('\t')[3])*100 #in 10ms
			diff=np.round(accumulator+end-start)
			accumulator=(accumulator+end-start)-diff
			for j in np.arange(diff):
				if i.split('\t')[4] == 'S':
					temp.append([1])
				else: 
					temp.append([0])
		dic[key] = temp
	return vad_key, dic 

if __name__ == '__main__':
	vad_import()	
