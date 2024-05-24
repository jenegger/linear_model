import numpy as np

# Example 3D array
arr = np.random.rand(2, 3, 4)  # Shape (2, 3, 4)
print("Starting array:", arr)

# Determine the shape of the zeros array
m, n, p = arr.shape
additional_zeros = 2  # Number of zeros slices to add in the second dimension

# Create the zeros array with the desired shape
zeros_array = np.zeros((m, additional_zeros, p))

# Concatenate along the second dimension (axis=1)
result = np.concatenate((arr, zeros_array), axis=1)

print("Original array shape:", arr.shape)
print("Zeros array shape:", zeros_array.shape)
print("Resulting array shape:", result.shape)
print("Resulting array:",result)

