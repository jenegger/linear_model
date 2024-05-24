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
		#print("combinations...")
		#print(some)
		#combination.append(itertools.combinations(lst, r))
	return combination

def permute_second_axis(arr):
    np.random.shuffle(arr)
    return arr



#1) Read from file
my_data = genfromtxt('data_stream_2121.txt', delimiter=',')
#normalize data
my_data[:,4] =(my_data[:,4]-np.min(my_data[:,4]))/(np.max(my_data[:,4])-np.min(my_data[:,4]))
my_data[:,1] =(my_data[:,1]-np.min(my_data[:,1]))/(np.max(my_data[:,1])-np.min(my_data[:,1]))
array_unique_events = np.unique(my_data[:,0])
size_of_unique_events = array_unique_events.shape[0];
modulo_val = size_of_unique_events % 3
if (modulo_val == 0):
	print("hello")
if (modulo_val == 1):
	print("modulo is 1")
	array_unique_events = array_unique_events[:-1]
if (modulo_val == 2):
	print("modulo is 2")
	array_unique_events = array_unique_events[:-2]
selected_hits = np.empty(0)
print(array_unique_events)
eventnumber = array_unique_events.shape[0]/3
print("number of events in here...")
print(eventnumber)
print(int(eventnumber))
selected_hits = np.random.choice(array_unique_events,3*int(eventnumber),replace=False)
selected_hits = np.resize(selected_hits,(int(eventnumber),3))
print(selected_hits)
#2)start for loop to select 3 random events
#for hits in selected_hits:
for hits in selected_hits[:5]:
	print(hits.shape)
	print(type(hits))
	print(type(hits[0]))
	print(hits[0].shape)
	nr_subevent1 = int(hits[0])
	nr_subevent2 = int(hits[1])
	nr_subevent3 = int(hits[2])
	subevent1 = my_data[my_data[:,0] == nr_subevent1]
	subevent2 = my_data[my_data[:,0] == nr_subevent2]
	subevent3 = my_data[my_data[:,0] == nr_subevent3]
	list_of_subev = [subevent1,subevent2,subevent3]

	#print("----start of new subevent----")
	#print(subevent1)
	#print(subevent1.shape)
	#print("----END of subevent----")
	full_event = np.concatenate((subevent1,subevent2,subevent3),axis=0)
	#print(subevent1.shape,subevent2.shape,subevent3.shape)
	#print(full_event)
	#print(full_event.shape)
	full_event_list = full_event.tolist()

	print(full_event_list)
	all_combinations = get_combinations(full_event_list)
	print(all_combinations)
	print(type(all_combinations))
	print (len(all_combinations))
	mask_list = []
	max_length_comb = 0;
	for elem in all_combinations:
		np_elem = np.array(elem)
		np_elem = np.reshape(np_elem,(-1,5))
		if (np_elem.shape[0] > max_length_comb):
			max_length_comb = np_elem.shape[0]
		print(np_elem)
		print(np_elem.shape)
		mask_val = 0
		for i in list_of_subev:
			mask_val += is_subarray(i,np_elem)
			print("check mask making----------------------")
			print("true array")
			print(i)
			print("comb array")
			print(np_elem)
			print(mask_val)
			print("END mask making------------------------")
		mask_list.append(mask_val)
	print(mask_list)		
	print ("this is the size of mask_list")
	print(len(mask_list))

	#now padding of the combinations to get subarrays with same length
	print(type(all_combinations))
	test= [np.array(sublist) for sublist in all_combinations]
	for elem in test:
		arr_elem = np.array(elem)
		arr_elem = np.reshape(elem,(-1,5))
		
	padded_arr  = [np.append(subarr,np.zeros(((max_length_comb - subarr.shape[0]),5)),axis=0) for subarr in test]

	print("this is the final numpy array")
	np_arr = np.asarray(padded_arr)
	#permute rows in the subarrays
	for x in np_arr:
		np.random.shuffle(x)
	#np_arr = np.apply_along_axis(permute_second_axis, axis=1, arr=np_arr)
	print(np_arr)
	print(np_arr.shape)
	
	#combinations = list(itertools.product(full_event))
	#print(combinations)


a2D = np.array([[1, 1.5,100], [2, 2.2,110],[3, 0.511,95]])
print ("shape of a2D")
print (a2D.shape)
list_a2D = a2D.tolist()
print("this is the list")
print(list_a2D)
my_comb = get_combinations(list_a2D)
print("this is the type of my_comb")
print(type(my_comb))

test= [np.array(sublist) for sublist in my_comb]
print(test)
print(test[0].shape)
print(test[4].shape)
print(type(test[4]))

#check if we can find same elements
for t_elem in test:
	for c_elem in a2D:
		arr_t_elem = np.array(t_elem)
		arr_c_elem = np.reshape(c_elem,(-1,3))
		print (arr_t_elem.shape)
		print(arr_c_elem.shape)
		if(arr_t_elem.shape[0] > arr_c_elem.shape[0]):
			energy = 0
			continue
		if(arr_t_elem.shape[0] <= arr_c_elem.shape[0]):
			print ("hello")
			for row in arr_t_elem:
				values = (np.all(arr_c_elem == row, axis=0))
				print(arr_c_elem)
				print (row)
				print(values)
		


max_length = max(subarr.shape[0] for subarr in test)
print("this is max length")
print(max_length)
padded_arr  = [np.append(subarr,np.zeros(((max_length - subarr.shape[0]),3)),axis=0) for subarr in test]
print("this is the final numpy array")
np_arr = np.asarray(padded_arr)
print(np_arr)
print(np_arr.shape)


#print (padded_arr)
#print (padded_arr.shape)
#print(my_comb)
#max_length = max(len(sublist) for sublist in my_comb)
#padded_lists = [sublist + [0]*(max_length - len(sublist)) for sublist in my_comb]
#array_with_padding = np.array(padded_lists)
#print (array_with_padding)
#
#
#arr_comb = np.asarray(my_comb)
#print(arr_comb)
#print (arr_comb.shape[0])
#print("these are now the combinations...")
#for comb in arr_comb:
#	print(comb)
#	arr_ = np.asarray(comb)
#	print(arr_.shape)
#
#print(arr_comb[1])

#3)make all different possible combinations
	

#4)check if they belong together

#5)inflate data so that all combinations have same size
