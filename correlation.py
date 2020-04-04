import csv, argparse
import numpy as np
from scipy.stats import pearsonr
from collections import defaultdict as ddict

# calculate pearson correlation for local predictions
def getCorrelationMicro(results):
	pearson_list = []
	with open(results, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			target      = np.array(list(map(float, row[3].strip('"[').strip(']"').split(','))))
			predicted   = np.array(list(map(float, row[4].strip('"[').strip(']"').split(','))))
			pearson_list.append(pearsonr(target, predicted)[0])
	return round(np.mean(pearson_list), 3)


# calculate pearson correlation for global predictions
def getCorrelationMacro(results):
	protein_map_target      = ddict(list)
	protein_map_predicted   = ddict(list)
	with open(results, 'r') as f:
		reader = csv.reader(f)
		for row in reader:
			protein     = row[0].split('_')[0]
			target      = float(row[1])
			predicted   = float(row[2])
			protein_map_target[protein].append(target)
			protein_map_predicted[protein].append(predicted)

	all_pearsons = []
	for k, v in protein_map_target.items():
		tgt     = v
		pred    = protein_map_predicted[k]
		pearson = pearsonr(np.array(tgt), np.array(pred))[0]
		all_pearsons.append(pearson)
	return round(np.mean(all_pearsons), 3)

parser = argparse.ArgumentParser(description='parser')
parser.add_argument('-file', help='The path to test_results.csv file')
args = parser.parse_args()

print('Local Pearson: ', getCorrelationMicro(args.file))
print('Global Pearson: ', getCorrelationMacro(args.file))
