##-----------------------------------------------------------------------------
##  Import
##-----------------------------------------------------------------------------
import argparse, os
from glob import glob
from tqdm import tqdm
from time import time
from scipy.io import savemat
from multiprocessing import cpu_count, Pool, Process, freeze_support
from fnc.extractFeature import extractFeature


#------------------------------------------------------------------------------
#	Argument parsing
#------------------------------------------------------------------------------


start = time()

parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", type=str, default="../CASIA1/",
					help="Path to the directory containing CASIA1 images.")

parser.add_argument("--temp_dir", type=str, default="./templates/CASIA1/",
					help="Path to the directory containing templates.")

parser.add_argument("--n_cores", type=int, default=cpu_count(),
					help="Number of cores used for enrolling template.")

args = parser.parse_args()


##-----------------------------------------------------------------------------
##  Pool function
##-----------------------------------------------------------------------------
def pool_func(file):
	template, mask, _ = extractFeature(file, use_multiprocess=False)
	basename = os.path.basename(file)
	out_file = os.path.join(args.temp_dir, "%s.mat" % (basename))
	savemat(out_file, mdict={'template': template, 'mask': mask}, appendmat=True)

def main():
	files = glob(os.path.join(args.data_dir, "*.jpg"))
	n_files = len(files)
	print("Number of files for enrolling:", n_files)

		# Parallel pools to enroll templates
	print("Start enrolling...")
	pools = Pool(processes=args.n_cores)
	for _ in tqdm(pools.imap_unordered(pool_func, files), total=n_files):
		pass
	pools.close()
	pools.join()

	end = time()
	print('\n>>> Enrollment time: {} [s]\n'.format(end-start))
	return 0

##-----------------------------------------------------------------------------
##  Execution
##-----------------------------------------------------------------------------
if not os.path.exists(args.temp_dir):
	print("makedirs", args.temp_dir)
	os.makedirs(args.temp_dir)

# Get list of files for enrolling template, just "xxx_1_x.jpg" files are selected

if __name__ == '__main__':
	main()
# Check the existence of temp_dir

