import sys, time, csv, os, random, math, argparse
import numpy as np
import torch
from tqdm import tqdm
from collections import OrderedDict
from arguments import buildParser

from model import ProteinGCN
from data import ProteinDataset, collate_pool, get_train_val_test_loader
from utils import AverageMeter, Normalizer, Logger, count_parameters, randomSeed, clearCache
import config as cfg


def main():
	global args, best_error_global, best_error_local, savepath, dataset

	parser  = buildParser()
	args    = parser.parse_args()

	print('Torch Device being used: ', cfg.device)

	# create the savepath
	savepath = args.save_dir + str(args.name) + '/'
	if not os.path.exists(savepath):
		os.makedirs(savepath)

	# Writes to file and also to terminal
	sys.stdout = Logger(savepath)
	print(vars(args))

	best_error_global, best_error_local = 1e10, 1e10

	randomSeed(args.seed)

	# create train/val/test dataset separately
	assert os.path.exists(args.protein_dir), '{} does not exist!'.format(args.protein_dir)
	all_dirs    = [d for d in os.listdir(args.protein_dir) if not d.startswith('.DS_Store')]
	dir_len     = len(all_dirs)
	indices     = list(range(dir_len))
	random.shuffle(indices)

	train_size  = math.floor(args.train * dir_len)
	val_size    = math.floor(args.val * dir_len)
	test_size   = math.floor(args.test * dir_len)

	if val_size == 0:
		print('No protein directory given for validation!! Please recheck the split ratios, ignore if this is intended.')
	if test_size == 0:
		print('No protein directory given for testing!! Please recheck the split ratios, ignore if this is intended.')


	test_dirs   = all_dirs[:test_size]
	train_dirs  = all_dirs[test_size:test_size + train_size]
	val_dirs   	= all_dirs[test_size + train_size:test_size + train_size + val_size]
	print('Testing on {} protein directories:'.format(len(test_dirs)))

	dataset = ProteinDataset(args.pkl_dir, args.id_prop, args.atom_init, random_seed=args.seed)

	print('Dataset length: ', len(dataset))

	# load all model args from pretrained model
	if args.pretrained is not None and os.path.isfile(args.pretrained):
		print("=> loading model params '{}'".format(args.pretrained))
		model_checkpoint    = torch.load(args.pretrained, map_location=lambda storage, loc: storage)
		model_args          = argparse.Namespace(**model_checkpoint['args'])
		# override all args value with model_args
		args.h_a            = model_args.h_a
		args.h_g            = model_args.h_g
		args.n_conv         = model_args.n_conv
		args.random_seed    = model_args.seed
		args.lr             = model_args.lr

		print("=> loaded model params '{}'".format(args.pretrained))
	else:
		print("=> no model params found at '{}'".format(args.pretrained))

	# build model
	kwargs = {
		'pkl_dir'       : args.pkl_dir,         # Root directory for data
		'atom_init'     : args.atom_init,       # Atom Init filename
		'h_a'           : args.h_a,             # Dim of the hidden atom embedding learnt
		'h_g'           : args.h_g,             # Dim of the hidden graph embedding after pooling
		'n_conv'        : args.n_conv,          # Number of GCN layers

		'random_seed'   : args.seed,            # Seed to fix the simulation
		'lr'            : args.lr,              # Learning rate for optimizer
	}

	structures, _, _    = dataset[0]
	h_b                 = structures[1].shape[-1]
	kwargs['h_b']       = h_b                       # Dim of the bond embedding initialization

	# Use DataParallel for faster training
	print("Let's use", torch.cuda.device_count(), "GPUs and Data Parallel Model.")
	model = ProteinGCN(**kwargs)
	model = torch.nn.DataParallel(model)
	model.cuda()

	print('Trainable Model Parameters: ', count_parameters(model))

	# Create dataloader to iterate through the dataset in batches
	train_loader, val_loader, test_loader = get_train_val_test_loader(dataset, train_dirs, val_dirs, test_dirs,
																		  collate_fn    = collate_pool,
																		  num_workers   = args.workers,
																		  batch_size    = args.batch_size,
																		  pin_memory    = False)

	try:
		print('Training data    : ', len(train_loader.sampler))
		print('Validation data  : ', len(val_loader.sampler))
		print('Testing data     : ', len(test_loader.sampler))
	except Exception as e:
		# sometimes test may not be defined
		print('\nException Cause: {}'.format(e.args[0]))

	# obtain target value normalizer
	if len(dataset) < args.avg_sample:  sample_data_list = [dataset[i] for i in tqdm(range(len(dataset)))]
	else:                               sample_data_list = [dataset[i] for i in tqdm(random.sample(range(len(dataset)), args.avg_sample))]

	_, _, sample_target = collate_pool(sample_data_list)
	normalizer_global   = Normalizer(sample_target[0])
	normalizer_local    = Normalizer(torch.tensor([0.0]))
	normalizer_local    = Normalizer(sample_target[1])

	# load the model state dict from given pretrained model
	if args.pretrained is not None and os.path.isfile(args.pretrained):
		print("=> loading model '{}'".format(args.pretrained))
		checkpoint = torch.load(args.pretrained, map_location=lambda storage, loc: storage)

		print('Best error global: ', checkpoint['best_error_global'])
		print('Best error local: ', checkpoint['best_error_local'])

		best_error_global = checkpoint['best_error_global']
		best_error_local  = checkpoint['best_error_local']

		model.module.load_state_dict(checkpoint['state_dict'])
		model.module.optimizer.load_state_dict(checkpoint['optimizer'])
		normalizer_local.load_state_dict(checkpoint['normalizer_local'])
		normalizer_global.load_state_dict(checkpoint['normalizer_global'])
	else:
		print("=> no model found at '{}'".format(args.pretrained))

	# Main training loop
	for epoch in range(args.epochs):
		# Training
		[train_error_global, train_error_local, train_loss] = trainModel(train_loader, model, normalizer_global, normalizer_local, epoch=epoch)
		# Validation
		[val_error_global, val_error_local, val_loss]       = trainModel(val_loader, model, normalizer_global, normalizer_local, epoch=epoch, evaluation=True)

		# check for error overflow
		if (val_error_global != val_error_global) or (val_error_local != val_error_local):
			print('Exit due to NaN')
			sys.exit(1)

		# remember the best error and possibly save checkpoint
		is_best             = val_error_global < best_error_global
		best_error_global   = min(val_error_global, best_error_global)
		best_error_local    = val_error_local

		# save best model
		if args.save_checkpoints:
			model.module.save({
				'epoch'             : epoch,
				'state_dict'        : model.module.state_dict(),
				'best_error_global' : best_error_global,
				'best_error_local'  : best_error_local,
				'optimizer'         : model.module.optimizer.state_dict(),
				'normalizer_global' : normalizer_global.state_dict(),
				'normalizer_local'  : normalizer_local.state_dict(),
				'args'              : vars(args)
			}, is_best, savepath)

	# test best model using saved checkpoints
	if args.save_checkpoints and len(test_loader):
		print('---------Evaluate Model on Test Set---------------')
		# this try/except allows the code to test on the go or by defining a pretrained path separately
		try:
			best_checkpoint = torch.load(savepath + 'model_best.pth.tar')
		except Exception as e:
			best_checkpoint = torch.load(args.pretrained)

		model.module.load_state_dict(best_checkpoint['state_dict'])
		[test_error_global, test_error_local, test_loss] = trainModel(test_loader, model, normalizer_global, normalizer_local, testing=True)


def trainModel(data_loader, model, normalizer_global, normalizer_local, epoch=None, evaluation=False, testing=False):
	"""
	The function to train/test the model for one epoch. Also, writes the test results to a file 'test_results.csv' in the end

	Parameters
	----------
	data_loader         : The data iterator to generate batches
	model               : The model to train
	normalizer_global   : The normalizer for global gdt targets
	normalizer_local    : The normalizer for local lddt targets
	epoch               : The current epoch
	evaluation          : (bool) Denotes if the model is in eval mode (True for both testing and validation)
	testing             : (bool) Denotes if the model is in test mode (True only while testing)

	Returns
	-------
	avg_errors_global   : The average global MAE error
	avg_errors_local    : The average local MAE error
	losses              : The average MSE loss
	"""
	batch_time          = AverageMeter()
	data_time           = AverageMeter()
	losses              = AverageMeter()
	avg_errors_global   = AverageMeter()
	avg_errors_local    = AverageMeter()

	# placeholders to store results to write to file
	if testing:
		test_targets_global = []
		test_preds_global   = []
		test_targets_local  = []
		test_preds_local    = []
		test_cif_ids        = []
		test_amino_crystal  = []

	end = time.time()

	for protein_batch_iter, (input_data, batch_data, target_tuples) in enumerate(data_loader):
		batch_protein_ids   = batch_data[0]
		batch_amino_crystal = batch_data[1]
		batch_size          = len(batch_protein_ids)

		# measure data loading time
		data_time.update(time.time() - end)

		# move inputs and targets to cuda
		input_var, target_var = getInputs(input_data, target_tuples, normalizer_global, normalizer_local)

		if not evaluation and not testing:
			# Switch to train mode
			model.train()

			out = model(input_var)
			out = model.module.mask_remove(out)
			assert out[1].shape[0]  == target_var[1].shape[0] ,  "Predicted Outputs Amino & Target Outputs Amino don't match"
			model.module.fit(out, target_var, batch_protein_ids)
		else:
			# evaluate one iteration
			with torch.no_grad():
				# Switch to evaluation mode
				model.eval()
				predicted = model(input_var)
				predicted = model.module.mask_remove(predicted)
				assert predicted[1].shape[0]  == target_var[1].shape[0] ,  "Predicted Outputs Amino & Target Outputs Amino don't match"
				model.module.fit(predicted, target_var, batch_protein_ids, pred=True)

		# Calculate the accuracy between the denormalized values
		model.module.accuracy[0] = model.module.accuracy[0] * normalizer_global.std
		model.module.accuracy[1] = model.module.accuracy[1] * normalizer_local.std

		# measure accuracy and record loss
		losses.update(model.module.loss.item(), batch_size)
		avg_errors_global.update(model.module.accuracy[0].item(), batch_size)
		avg_errors_local.update(model.module.accuracy[1].item(), batch_size)

		# Collect all the results that needs to be written to file
		if testing and batch_size != 1:
				test_pred_global    = normalizer_global.denorm(model.module.outputs[0].data).squeeze().tolist()
				test_target_global  = target_tuples[0].squeeze()
				test_preds_global   += test_pred_global
				test_targets_global += test_target_global.tolist()

				test_amino_crystal  += batch_amino_crystal.tolist()
				test_pred_local     = normalizer_local.denorm(model.module.outputs[1].data).squeeze().tolist()
				test_target_local   = target_tuples[1].squeeze().tolist()

				res1, res2 = OrderedDict(), OrderedDict()
				for i, idx in enumerate(batch_amino_crystal):
					if idx not in res1: res1[idx] = []
					if idx not in res2: res2[idx] = []
					res1[idx].append(test_target_local[i])
					res2[idx].append(test_pred_local[i])

				test_target_local   = [v for _, v in res1.items()]
				test_pred_local     = [v for _, v in res2.items()]

				test_preds_local    += test_pred_local
				test_targets_local  += test_target_local
				test_cif_ids        += batch_protein_ids

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# print progress between steps
		if protein_batch_iter % args.print_freq == 0:
			if evaluation or testing:
				print('Test: [{0}][{1}]/{2}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'ERRG {avg_errors_global.val:.3f} ({avg_errors_global.avg:.3f})\t'
					  'ERRL {avg_errors_local.val:.3f} ({avg_errors_local.avg:.3f})'.format(
					epoch, protein_batch_iter, len(data_loader), batch_time=batch_time, loss=losses,
					avg_errors_global=avg_errors_global, avg_errors_local=avg_errors_local))
			else:
				print('Epoch: [{0}][{1}]/{2}\t'
					  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
					  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
					  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
					  'ERRG {avg_errors_global.val:.3f} ({avg_errors_global.avg:.3f})\t'
					  'ERRL {avg_errors_local.val:.3f} ({avg_errors_local.avg:.3f})'.format(
					epoch, protein_batch_iter, len(data_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, avg_errors_global=avg_errors_global,
					avg_errors_local=avg_errors_local))

		if protein_batch_iter % args.print_freq == 0:
			clearCache()

	# write results to file
	if testing:
		star_label = '**'
		with open(savepath + 'test_results.csv', 'w') as f:
			writer = csv.writer(f)
			for cif_id, targets_global, preds_global, targets_local, preds_local in zip(test_cif_ids,
																						test_targets_global,
																						test_preds_global,
																						test_targets_local,
																						test_preds_local):
				writer.writerow((cif_id, targets_global, preds_global, targets_local, preds_local))
	elif evaluation:
		star_label = '*'
	else:
		star_label = '##'

	print(' {star} ERRG {avg_errors_global.avg:.3f} ERRL {avg_errors_local.avg:.3f} LOSS {avg_loss.avg:.3f}'.format(
			star=star_label, avg_errors_global=avg_errors_global, avg_errors_local=avg_errors_local, avg_loss=losses))

	return avg_errors_global.avg, avg_errors_local.avg, losses.avg


def getInputs(inputs, target_tuples, normalizer_global, normalizer_local):
	"""Move inputs and targets to cuda"""

	input_var               = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda(), inputs[4].cuda(), inputs[5].cuda()]
	target_global           = target_tuples[0].cuda()
	target_local            = target_tuples[1].cuda()
	target_global_normed    = normalizer_global.norm(target_global)
	target_local_normed     = normalizer_local.norm(target_local)
	target_var              = [target_global_normed.cuda(), target_local_normed.cuda()]

	return input_var, target_var


if __name__ == '__main__':
	start = time.time()
	main()
	print('Time taken: ', time.time() - start)
