from torch.utils.data import DataLoader
import torch.optim as optim
from dataloader import *
from model import *


def main():

	# create dataloader
	dataset = COMDataset

	# load model
	Model = Net()

	# create optimizer
	optimizer = optim.Adam(Model.parameters(), lr=0.001)

	# create checkpoint directory
	checkpoint_dir = "checkpoint"

	# start training
	trainer = Trainer(COMDataset, Model, optimizerm, checkpoint_dir, batch_size = 1)

	num_epochs = 100
	trainer.train(num_epochs)

if __name__ = "__main__":
	main()