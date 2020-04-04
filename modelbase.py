import torch
import torch.nn as nn


class ModelBase(nn.Module):
	"""The base class which every model inherits"""
	def __init__(self, **kwargs):
		super(ModelBase, self).__init__()

		# Build the model here. This is required because to initialize the optimiser the model parameters
		# need to be available beforehand
		self.build(**kwargs)

		self.inputs     = None
		self.targets    = None
		self.outputs    = None
		self.loss       = 0
		self.accuracy   = 0
		self.optimizer  = None

	def _build(self, **kwargs):
		raise NotImplementedError

	def build(self, **kwargs):
		"""Wrapper for _build()"""
		self._build(**kwargs)

	def _forward(self, inputs):
		raise NotImplementedError

	def forward(self, inputs):
		"""Wrapper for _forward()"""
		self.inputs = inputs
		self.outputs = self._forward(self.inputs)

	def fit(self, inputs, targets):
		"""Train the model for given inputs"""

		# Switch to train mode - useful for Dropout claculations etc.
		self.train()

		self.targets = targets
		self.forward(inputs)

		self._loss()
		self._accuracy()

		self.optimizer.zero_grad()
		self.loss.backward()
		self.optimizer.step()

	def _loss(self):
		raise NotImplementedError

	def _accuracy(self):
		raise NotImplementedError

	def predict(self, inputs, targets):
		# Switch to evaluation mode - useful for Dropout claculations etc.
		self.eval()

		self.forward(inputs)
		self.targets = targets
		self._loss()
		self._accuracy()

	def save(self, state, filename):
		"""Saves the given state (as a dictionary) in given filename"""
		torch.save(state, filename)

	def load(self):
		pass
