import torch
from loss import loss
from utils import *

class Trainer():

	def __init__(self, COMdataset, Model, optimizer, checkpoint_dir, batch_size):
		self.COMdataset = COMdataset
		self.model = self.Model
		# optimizer to perform backpropagation
		self.optimizer = optimizer
		# directory to save the model
		self.checkpoint_dir = checkpoint_dir
		# gpu or cpu
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		self.model.to(self.device)
		# split the dataset into training, validating, and testing sets
		self.train_val_test = [0.7,0.1,0.2]
		self.batch_size = batch_size
		# loss
		self.criterion = loss()
		self.save_per_epochs = 1


		 # Randomly split dataset into training, validation and test sets
        train_size = int(self.train_val_test[0]*len(self.COMdataset))
        val_size = int(self.train_val_test[1]*len(self.COMdataset))
        test_size = len(self.COMdataset) - train_size - val_size

        # random_split() is from PyTorch
        train_DataSet, val_DataSet, test_DataSet = random_split(self.COMdataset, [train_size, val_size, test_size])

        self.train_DataLoader = DataLoader(train_DataSet, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_DataLoader = DataLoader(val_DataSet, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.test_DataLoader = DataLoader(test_DataSet, batch_size=self.batch_size, shuffle=True, num_workers=4)

	def train(self, num_epochs):
		for epoch in range(1, num_epochs+1):

			# set model for training
			# model.train() for training
			# model.eval() for evaluating
			self.model.train()

			for batch_idx, data, y in enumerate(self.train_DataLoader):

				data = data.to(self.device)
				# let the gradient of the weights be zero
				self.optimizer.zero_grad()

				# feed into the model
				output, latent = self.model(data)

				# compute loss
				total_loss = self.criterion(data, output, latent, y)

				# backpropagate
				total_loss.backward()

			# Eval: get train, val and test accuracies
            training_acc, training_loss = self._compute_metrics(self.train_DataLoader)
            validate_acc, validation_loss = self._compute_metrics(self.val_DataLoader)
            sys.stdout.write(" train acc: {:.3f}, valid acc: {:.3f}".format(
                training_acc, validate_acc))
            if epoch % self.save_per_epochs == 0:
            	if validate_acc > best_validate_acc:
	                best_validate_acc = validate_acc
	                saved_model_testing_acc = testing_acc
	                self._save_checkpoint(epoch, training_loss)

  def _compute_metrics(self, input_DataLoader):
    # torch.no_grad() saves memory and increases computation speed
    with torch.no_grad():
        self.model.eval()
        total_metrics = 0
        total_loss = 0
        for batch_idx, (x, y) in enumerate(input_DataLoader):
         pass
        return total_metrics / len(input_DataLoader.dataset),  total_loss / len(input_DataLoader.dataset)









