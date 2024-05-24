import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_shaping import create_data
from my_first_model import TinyModel

class CustomDataset(Dataset):
	def __init__(self):
		self.d_set, self.target = create_data()
		assert  len(self.d_set) == len(self.target)
	
	def __len__(self):
		return len(self.d_set)
	
	def __getitem__(self, idx):
		event = self.d_set[idx]
		mask = self.target[idx]
		return event,mask


dataset = CustomDataset()
#print("---------------HERE STARTS THE DATASET PRINTING----------------")
#for item in dataset:
#	print("THIS IS ELEMENT OF DATASET")
#	print(item)
#	print(type(item[0]))
#	print(item[0].shape)
#	print(type(item[1]))
#	print(len(item[1]))
#print("---------------HERE STARTS THE DATASET PRINTING----------------")
#dloader = DataLoader(dataset,batch_size=4,shuffle=False)
#for batch in dloader:
#	print(batch)

def dynamic_length_collate(batch):
	#print("this is type of batch:",type(batch))
	#print("this is type of first entry in batch:",type(batch[0]))
	#print("this is type of first entry in batchin the tuple:",type(batch[0][0]))
	list_of_lists = list(map(list, batch))
	#print(list_of_lists[0][0])
	#print(type(list_of_lists[0][0]))
	for sublist in list_of_lists:
		print("shape of the combination batches")
		print(sublist[0].shape)
		print("length of the masks:")
		print(len(sublist[1]))
	in_data = [sublist[0] for sublist in list_of_lists]
	in_target = [sublist[1] for sublist in list_of_lists]
	nr_comb = max(item.shape[0] for item in in_data)
	nr_max_hits = max(item.shape[1] for item in in_data)
	print("max number of combinations:",nr_comb)
	print("max number of hits:",nr_max_hits)
	#nr_comb = in_data.shape[0]
	#nr_max_hits = in_data.shape[1]
	#nr_comb = max(item.shape[0] for item in batch[0])
	#nr_max_hits = max(item.shape[1] for item in batch[0])
	out_data = []
	out_target = []
	for in_data,in_target in batch:
		pad_comb = nr_comb - in_data.shape[0]
		pad_hits = nr_max_hits - in_data.shape[1]
		zeros_array = np.zeros((in_data.shape[0],pad_hits, 5))
		result_data = np.concatenate((in_data, zeros_array), axis=1)
		zeros_array_comb = np.zeros((pad_comb,nr_max_hits,5))
		result_data = np.concatenate((result_data,zeros_array_comb), axis =0)
		zeros_target = np.zeros(pad_comb)
		np_in_target = np.array(in_target)
		result_target = np.concatenate((np_in_target,zeros_target))
		out_data.append(result_data)
		out_target.append(result_target)
	print("this is the shape of out_data",len(out_data), out_data[0].shape)
	#return out_data, out_target
	np_out_data = np.array(out_data)
	np_out_target = np.array(out_target)
	return torch.from_numpy(np_out_data), torch.from_numpy(np_out_target)
dloader = DataLoader(dataset,batch_size=8,shuffle=False,collate_fn=dynamic_length_collate)
for batch in dloader:
	print("hello")
	print(type(batch))
	print(len(batch))
	print(type(batch[0]))
	print(type(batch[1]))
	print(batch[0].size())
	print(batch[1].size())
	pred = TinyModel() 

#	max_len = max(len(item) for item in batch)
#	for  item in batch:
#		pad_len = max_len - len(item)
#		batch_data.append(...)
#		batch_target.append(...)
#	return batch_data,batch_target

	
