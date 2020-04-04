from __future__ import print_function, division
import os, functools, math, csv, random, pickle, json
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
from collections import defaultdict as ddict


def get_train_val_test_loader(dataset, train_dirs, val_dirs, test_dirs, collate_fn=default_collate, batch_size=64, num_workers=1, pin_memory=False, predict=False):
	"""
	Utility function for dividing a dataset to batches

	Parameters
	----------
	dataset: torch.utils.data.Dataset
		The full dataset to be divided.
	train/val/test_dirs: list
		Only consider proteins from the specified directories in the split
	batch_size: int
		Batch size for training
	num_workers: int
	pin_memory: bool
		Useful for GPU runs
	predict: bool
		If true, just return test set

	Returns
	-------
	train_loader: torch.utils.data.DataLoader
		DataLoader that random samples the training data.
	(val_loader): torch.utils.data.DataLoader
		DataLoader that random samples the validation data, returns if
		return_val=True
	(test_loader): torch.utils.data.DataLoader
		DataLoader that random samples the test data, returns if
		return_test=True.
	"""

	if not predict:
		train_indices   = [i for i, row in enumerate(dataset.id_prop_data) if row[0].split('_')[0] in train_dirs]
		val_indices     = [i for i, row in enumerate(dataset.id_prop_data) if row[0].split('_')[0] in val_dirs]
		test_indices    = [i for i, row in enumerate(dataset.id_prop_data) if row[0].split('_')[0] in test_dirs]

		random.shuffle(train_indices)
		random.shuffle(val_indices)
		random.shuffle(test_indices)

		# Sample elements randomly from a given list of indices, without replacement.
		train_sampler   = SubsetRandomSampler(train_indices)
		val_sampler     = SubsetRandomSampler(val_indices)
		test_sampler    = SubsetRandomSampler(test_indices)

		train_loader    = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
		val_loader      = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
		test_loader     = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
		return train_loader, val_loader, test_loader
	else:
		test_indices    = [i for i, row in enumerate(dataset.id_prop_data) if row[0].split('_')[0] in test_dirs]
		random.shuffle(test_indices)
		test_sampler    = SubsetRandomSampler(test_indices)
		test_loader     = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
		return test_loader


def collate_pool(dataset_list):
	N   = max([x[0][0].size(0) for x in dataset_list])  # max atoms
	A   = max([len(x[1][1]) for x in dataset_list])     # max amino in protein
	M   = dataset_list[0][0][1].size(1)                 # num neighbors are same for all so take the first value
	B   = len(dataset_list)                             # Batch size
	h_b = dataset_list[0][0][1].size(2)                 # Edge feature length

	final_protein_atom_fea = torch.zeros(B, N)
	final_nbr_fea          = torch.zeros(B, N, M, h_b)
	final_nbr_fea_idx      = torch.zeros(B, N, M, dtype=torch.long)
	final_atom_amino_idx   = torch.zeros(B, N)
	final_atom_mask        = torch.zeros(B, N)
	final_target           = torch.zeros(B, 1)
	final_amino_target     = []
	amino_base_idx         = 0

	batch_protein_ids, batch_amino_crystal, amino_crystal = [], [], 0
	for i, ((protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target, amino_target), protein_id) in enumerate(dataset_list):
		num_nodes                             = protein_atom_fea.size(0)
		num_amino                             = len(amino_target)
		final_protein_atom_fea[i][:num_nodes] = protein_atom_fea.squeeze()
		final_nbr_fea[i][:num_nodes]          = nbr_fea
		final_nbr_fea_idx[i][:num_nodes]      = nbr_fea_idx
		final_atom_amino_idx[i][:num_nodes]   = atom_amino_idx + amino_base_idx
		final_atom_amino_idx[i][num_nodes:]   = amino_base_idx
		amino_base_idx                       += torch.max(atom_amino_idx) + 1
		final_target[i]                       = target
		final_atom_mask[i][:num_nodes]        = 1
		final_amino_target.append(amino_target)

		batch_protein_ids.append(protein_id)
		batch_amino_crystal.append([amino_crystal for _ in range(len(amino_target))])
		amino_crystal += 1

	return (final_protein_atom_fea, final_nbr_fea, final_nbr_fea_idx, None, final_atom_amino_idx, final_atom_mask),\
			(batch_protein_ids, np.concatenate(batch_amino_crystal)), (final_target, torch.cat(final_amino_target))


class GaussianDistance(object):
	"""
	Expands the distance by Gaussian basis.

	Unit: angstrom
	"""
	def __init__(self, dmin, dmax, step, var=None):
		"""
		Parameters
		----------
		dmin: float
			Minimum interatomic distance
		dmax: float
			Maximum interatomic distance
		step: float
			Step size for the Gaussian filter
		"""
		assert dmin < dmax
		assert dmax - dmin > step
		self.filter = np.arange(dmin, dmax+step, step)
		if var is None:
			var = step
		self.var = var

	def expand(self, distances):
		"""
		Apply Gaussian distance filter to a numpy distance array

		Parameters
		----------
		distance: np.array shape n-d array
			A distance matrix of any shape

		Returns
		-------
		expanded_distance: shape (n+1)-d array
			Expanded distance matrix with the last dimension of length
			len(self.filter)
		"""
		return np.exp(-(distances[..., np.newaxis] - self.filter)**2 / self.var**2)


class AtomInitializer(object):
	"""
	Base class for intializing the vector representation for atoms.
	"""
	def __init__(self, atom_types):
		self.atom_types = set(atom_types)
		self._embedding = {}

	def get_atom_fea(self, atom_type):
		assert atom_type in self.atom_types
		return self._embedding[atom_type]

	def load_state_dict(self, state_dict):
		self._embedding = state_dict
		self.atom_types = set(self._embedding.keys())
		self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}

	def state_dict(self):
		return self._embedding

	def decode(self, idx):
		if not hasattr(self, '_decodedict'):
			self._decodedict = {idx: atom_type for atom_type, idx in self._embedding.items()}
		return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
	"""
	Initialize atom feature vectors using a JSON file, which is a python
	dictionary mapping from element number to a list representing the
	feature vector of the element.

	Parameters
	----------
	elem_embedding_file: str
		The path to the .json file
	"""
	def __init__(self, elem_embedding_file):
		with open(elem_embedding_file) as f:
			elem_embedding = json.load(f)
		elem_embedding = {key: value for key, value in elem_embedding.items()}
		atom_types = set(elem_embedding.keys())
		super(AtomCustomJSONInitializer, self).__init__(atom_types)
		counter = 0
		for key, _ in elem_embedding.items():
			self._embedding[key] = counter; counter += 1


class ProteinDataset(Dataset):
	"""
	The ProteinDataset dataset is a wrapper for a protein dataset where the protein structures
	are stored in the form of pkl files. The dataset should have the following
	directory structure:

	pkl_dir
	├── protein_id_prop.csv
	├── protein_atom_init.json
	├── id0.pkl
	├── id1.pkl
	├── ...

	protein_id_prop.csv: a CSV file with one column which recodes a
	unique ID for each protein along with the property value

	protein_atom_init.json: a JSON file that stores the initialization vector for each
	protein atom.

	ID.pkl: a pickle file that contains the follwing:
		protein_atom_fea: torch.Tensor shape (n_i, atom_fea_len)
		nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
		nbr_fea_idx: torch.LongTensor shape (n_i, M)
		protein_id: str or int

	Parameters
	----------
	pkl_dir: str
		The path to the pkl directory of the dataset
	id_prop_filename: str
		Name of the id_prop file to use for targets
	atom_init_filename: str
		Name of the atom_init file to use for atom embedding
	random_seed: int
		Random seed for shuffling the dataset

	Returns
	-------
	protein_atom_fea: torch.Tensor shape (n_i, atom_fea_len)
	nbr_fea         : torch.Tensor shape (n_i, M, nbr_fea_len)
	nbr_fea_idx     : torch.LongTensor shape (n_i, M)
	atom_amino_idx  : torch.LongTensor
	target          : torch.Tensor
	amino_target    : torch.Tensor
	protein_id      : str or int
	"""
	def __init__(self, pkl_dir, id_prop_filename, atom_init_filename, random_seed=123):
		assert os.path.exists(pkl_dir), '{} does not exist!'.format(pkl_dir)

		self.pkl_dir = pkl_dir
		id_prop_file = os.path.join(self.pkl_dir, id_prop_filename)
		assert os.path.exists(id_prop_file), '{} does not exist!'.format(id_prop_file)

		with open(id_prop_file) as f:
			reader = csv.reader(f)
			self.id_prop_data = [row for row in reader]
		random.seed(random_seed)
		random.shuffle(self.id_prop_data)

		protein_atom_init_file  = os.path.join(self.pkl_dir, atom_init_filename)
		assert os.path.exists(protein_atom_init_file), '{} does not exist!'.format(protein_atom_init_file)
		self.ari                = AtomCustomJSONInitializer(protein_atom_init_file)
		self.gdf                = GaussianDistance(dmin=0, dmax=15, step=0.4)

	def __len__(self):
		return len(self.id_prop_data)

	def __getitem__(self, idx):
		return self.get_idx(idx)

	def get_idx(self, idx):
		protein_id, target, amino_target = self.id_prop_data[idx]
		amino_target = [float(m.strip(" '").strip("'")) for m in amino_target.strip('"[').strip(']"').split(',')]

		with open(self.pkl_dir + protein_id + '.pkl', 'rb') as f:
			protein_atom_fea    = torch.Tensor(np.vstack([self.ari.get_atom_fea(atom) for atom in pickle.load(f)]))     # Atom features (here one-hot encoding is used)
			nbr_info            = pickle.load(f)                                                                        # Edge features for each atom in the graph
			nbr_fea_idx         = torch.LongTensor(pickle.load(f))                                                      # Edge connections that define the graph

			atom_amino_idx  = torch.LongTensor(pickle.load(f))  # Mapping that denotes which atom corresponds to which amino residue in the protein graph
			assert len(amino_target) == atom_amino_idx[-1] + 1  # (useful for calculating the amino level lddt score)

			protein_id          = pickle.load(f)
			nbr_fea             = torch.Tensor(np.concatenate([self.gdf.expand(nbr_info[:,:,0]), nbr_info[:,:,1:]], axis=2))    # Use Gaussian expansion for edge distance
			target              = torch.Tensor([float(target)])     # Global gdt score
			amino_target        = torch.Tensor(amino_target)        # Amino level lddt score

		return (protein_atom_fea, nbr_fea, nbr_fea_idx, atom_amino_idx), (target, amino_target), protein_id
