import numpy as np

def is_subarray(large_array, small_array):
	rows_large, cols_large = large_array.shape
	rows_small, cols_small = small_array.shape
	energy_val = 0
	#ene_sum = np.sum(large_array,axis=0)[1]
	ene_sum = np.sum(large_array,axis=0)[0]

	if cols_large != cols_small:
		return energy_val
	if rows_large < rows_small:
		return energy_val
	for j in range(rows_small):
		some_true_element = False
		for i in range(rows_large):
			if (np.array_equal(small_array[j,:],large_array[i,:])):
				some_true_element = True
				#energy_val += small_array[j,1]/ene_sum
				energy_val += small_array[j,0]/ene_sum
		#if (some_true_element == False):
		if (some_true_element  == False):
			return 0
	return energy_val

	#return False

# Example usage
large_array = np.array([[1, 2, 3], 
                        [4, 5, 6], 
                        [7, 8, 9], 
                        [10, 11, 12], 
                        [13, 14, 15]])

small_array = np.array([[1, 2, 3], 
                        [13, 14, 15]])

result = is_subarray(large_array, small_array)
print(result)
