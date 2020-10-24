import configargparse

def buildParser():
	parser = configargparse.ArgParser(default_config_files=['settings.conf'])

	# Data source
	parser.add('name',                                              help='Name of folder where result is saved')
	parser.add('--pkl_dir',     default='./data/pkl/',              help='Source directory for pkl files')
	parser.add('--protein_dir', default='./data/protein/',          help='Directory where all protein pdb files exist') # needed for inferring train/test sets
	parser.add('--save_dir',    default='./data/pkl/results/',      help='Destination directory for results')
	parser.add('--id_prop',     default='protein_id_prop.csv',      help='id_prop filename')
	parser.add('--atom_init',   default='protein_atom_init.json',   help='atom_init filename')
	parser.add('--pretrained',                                      help='Path to pretrained model')
	parser.add('--avg_sample',  default=500,                        help='Normalizer sample count for calculating mean and std of target', type=int)

	# Training setup
	parser.add('--seed',        default=1234,                       help='Seed for random number generation',   type=int)
	parser.add('--epochs',      default=100,                        help='Number of epochs',                    type=int)
	parser.add('--batch_size',  default=3,                          help='Batch size for training',             type=int)
	parser.add('--train',       default=0.5,                        help='Fraction of training data',           type=float)
	parser.add('--val',         default=0.25,                       help='Fraction of validation data',         type=float)
	parser.add('--test',        default=0.25,                       help='Fraction of test data',               type=float)
	parser.add('--testing',                                         help='If only testing the model',           action='store_true')

	# Optimizer setup
	parser.add('--lr',          default=0.001,                      help='Learning rate',                       type=float)

	# Model setup
	parser.add('--h_a',         default=64,                         help='Atom hidden embedding dimension',     type=int)
	parser.add('--h_g',         default=32,                         help='Graph hidden embedding dimension',    type=int)
	parser.add('--n_conv',      default=4,                          help='Number of convolution layers',        type=int)

	# Other features
	parser.add('--save_checkpoints',    default=True,               help='Stores checkpoints if true',                      action='store_true')
	parser.add('--print_freq',          default=10,                 help='Frequency of printing updates between epochs',    type=int)
	parser.add('--workers',             default=20,                 help='Number of workers for data loading',              type=int)

	return parser
