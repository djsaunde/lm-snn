import torch
import torch.nn


class Autoencoder(nn.Module):
	def __init__(self):
		super(Autoencoder, self).__init__()

		self.encoder = nn.Linear(784, 10)
		self.decoder = nn.Linear(10, 784)

	def forward(self, x):
