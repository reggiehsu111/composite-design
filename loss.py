import torch.nn as nn
import torch

class loss():
	def __init__(self):
		self.recon_loss = nn.BCELoss()
		self.a1, self.a2 = 1,1

	def __call__(self, data, output, latent, y):
		# compute for reconstruction loss
		recon = self.recon_loss(data,output)
		# compute latent loss
		latent_loss = (latent[0][0] - y)**2
		print("latent_loss: ", latent_loss)
		print(recon)
		return self.a1*recon+self.a2*latent_loss
		

if __name__ == '__main__':
	device = torch.device('cpu') 
	tensor = torch.tensor((), dtype=torch.float32)
	data = tensor.new_full(size=(1,1,32,32), fill_value=1).to(device)
	output = tensor.new_full(size=(1,1,32,32), fill_value=0.5).to(device)
	latent = tensor.new_full(size=(1,16),fill_value = 1).to(device)
	y = 4
	Loss = loss()
	total_loss = Loss(data,output,latent,y)
	print("total_loss: ", total_loss)