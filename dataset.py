from torch.utils.data import Dataset, DataLoader



class COMDataset(Dataset):
	def __init__(self, data_dir):
		pass

	def __getitem__(self, idx):
		pass

	def __len__(self):
		pass

if __name__ == '__main__':
	testdt = COMDataset('/data')

	# call getitem
	testdt[0]

	# call len
	len(COMDataset)

	COMdataloader =  DataLoader(testdt, batch_size=32, shuffle=True, num_workers=4)