#!/usr/bin/env python3
import itertools
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import math
from numpy import genfromtxt
from array_check import is_subarray

##some definitions
def get_combinations(lst):
	combination = []
	for r in range(1, len(lst) + 1):
		combination += itertools.combinations(lst, r)
		some = list(itertools.combinations(lst, r))
	return combination
def create_data():
	#1) Read from file
	my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
	#normalize data
	my_data[:,4] =(my_data[:,4]-np.min(my_data[:,4]))/(np.max(my_data[:,4])-np.min(my_data[:,4]))
	my_data[:,1] =(my_data[:,1]-np.min(my_data[:,1]))/(np.max(my_data[:,1])-np.min(my_data[:,1]))
	array_unique_events = np.unique(my_data[:,0])
	size_of_unique_events = array_unique_events.shape[0];
	modulo_val = size_of_unique_events % 3
	if (modulo_val == 0):
		print("modulo is 0")
	if (modulo_val == 1):
		print("modulo is 1")
		array_unique_events = array_unique_events[:-1]
	if (modulo_val == 2):
		print("modulo is 2")
		array_unique_events = array_unique_events[:-2]
	selected_hits = np.empty(0)
	eventnumber = array_unique_events.shape[0]/3
	selected_hits = np.random.choice(array_unique_events,3*int(eventnumber),replace=False)
	selected_hits = np.resize(selected_hits,(int(eventnumber),3))
	data_list =  []
	target_list = []
	for hits in selected_hits[:16]:
	#for hits in selected_hits:
		nr_subevent1 = int(hits[0])
		nr_subevent2 = int(hits[1])
		nr_subevent3 = int(hits[2])
		subevent1 = my_data[my_data[:,0] == nr_subevent1]
		subevent2 = my_data[my_data[:,0] == nr_subevent2]
		subevent3 = my_data[my_data[:,0] == nr_subevent3]
		subevent1 = np.delete(subevent1,0,1)
		subevent2 = np.delete(subevent2,0,1)
		subevent3 = np.delete(subevent3,0,1)
		list_of_subev = [subevent1,subevent2,subevent3]
	
		full_event = np.concatenate((subevent1,subevent2,subevent3),axis=0)
		full_event_list = full_event.tolist()
		all_combinations = get_combinations(full_event_list)
		mask_list = []
		max_length_comb = 0;
		for elem in all_combinations:
			np_elem = np.array(elem)
			#np_elem = np.reshape(np_elem,(-1,5))
			np_elem = np.reshape(np_elem,(-1,4))
			if (np_elem.shape[0] > max_length_comb):
				max_length_comb = np_elem.shape[0]
			mask_val = 0
			for i in list_of_subev:
				mask_val += is_subarray(i,np_elem)
			mask_list.append(mask_val)
	
		#now padding of the combinations to get subarrays with same length
		test= [np.array(sublist) for sublist in all_combinations]
		for elem in test:
			arr_elem = np.array(elem)
			#arr_elem = np.reshape(elem,(-1,5))
			arr_elem = np.reshape(elem,(-1,4))
			
		#padded_arr  = [np.append(subarr,np.zeros(((max_length_comb - subarr.shape[0]),5)),axis=0) for subarr in test]
		padded_arr  = [np.append(subarr,np.zeros(((max_length_comb - subarr.shape[0]),4)),axis=0) for subarr in test]
	
		np_arr = np.asarray(padded_arr)
		#permute rows in the subarrays
		for x in np_arr:
			np.random.shuffle(x)
		#print(np_arr)
		#print(mask_list)
		data_list.append(np_arr)
		target_list.append(mask_list)
	return data_list,target_list 

data,target = create_data()
