import sys
from subprocess import call
import torch
import config as cfg

class Logger(object):
	"""Writes both to file and terminal"""
	def __init__(self, savepath, mode='a'):
		self.terminal = sys.stdout
		self.log = open(savepath + 'logfile.log', mode)

	def write(self, message):
		self.terminal.write(message)
		self.log.write(message)

	def flush(self):
		# this flush method is needed for python 3 compatibility.
		# this handles the flush command by doing nothing.
		# you might want to specify some extra behavior here.
		pass


class Normalizer(object):
	"""Normalize a Tensor and restore it later. """
	def __init__(self, tensor):
		"""tensor is taken as a sample to calculate the mean and std"""
		self.mean = torch.mean(tensor).type(cfg.FloatTensor)
		self.std = torch.std(tensor).type(cfg.FloatTensor)

	def norm(self, tensor):
		if self.mean != self.mean or self.std != self.std:
			return tensor
		return (tensor - self.mean) / self.std

	def denorm(self, normed_tensor):
		if self.mean != self.mean or self.std != self.std:
			return normed_tensor
		return normed_tensor * self.std + self.mean

	def state_dict(self):
		return {'mean': self.mean,
				'std': self.std}

	def load_state_dict(self, state_dict):
		self.mean = state_dict['mean']
		self.std = state_dict['std']


class AverageMeter(object):
	"""
	Computes and stores the average and current value. Accomodates both numbers and tensors.
	If the input to be monitored is a tensor, also need the dimensions/shape of the tensor.
	Also, for tensors, it keeps a column wise count for average, sum etc.
	"""
	def __init__(self, is_tensor=False, dimensions=None):
		if is_tensor and dimensions is None:
			print('Bad definition of AverageMeter!')
			sys.exit(1)
		self.is_tensor = is_tensor
		self.dimensions = dimensions
		self.reset()

	def reset(self):
		self.count = 0
		if self.is_tensor:
			self.val = torch.zeros(self.dimensions, device=cfg.device)
			self.avg = torch.zeros(self.dimensions, device=cfg.device)
			self.sum = torch.zeros(self.dimensions, device=cfg.device)
		else:
			self.val = 0
			self.avg = 0
			self.sum = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def randomSeed(random_seed):
	"""Given a random seed, this will help reproduce results across runs"""
	if random_seed is not None:
		torch.manual_seed(random_seed)
		if torch.cuda.is_available():
			torch.cuda.manual_seed_all(random_seed)

def clearCache():
	torch.cuda.empty_cache()
