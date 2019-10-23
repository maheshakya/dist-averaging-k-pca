#Load dependencies
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.stats import ortho_group
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from sacred import Experiment
ex = Experiment('dist_kpca_fma')

@ex.config
def cfg():
  rs = 8

@ex.automain
def run(rs):
	def weighted_cov_sum(original, n, m, N, d, rng, k, n_vects):
	    UU_T = np.zeros((d,d))
	    UU_T_weighted = np.zeros((d, d))
	    UU_T_weighted_ = np.zeros((d, d))
	    for i in range(m):
	        inds = rng.choice(np.arange(N), n, replace=True)
	        A_i = original[:, inds]
	        X_i = np.dot(A_i, A_i.T)/n
	        U_i, Sig_i, VT_i = np.linalg.svd(X_i)
	        UU_T += np.dot(U_i[:, :k], U_i[:, :k].T)
	        UU_T_weighted += np.dot(U_i[:, :k], np.dot(np.diag(Sig_i[:k]), U_i[:, :k].T))        
	        UU_T_weighted_ += np.dot(U_i[:, :n_vects], np.dot(np.diag(Sig_i[:n_vects]), U_i[:, :n_vects].T))
	    return UU_T/float(m),UU_T_weighted/float(m), UU_T_weighted_/float(m)

    rng = np.random.RandomState(rs)


    #Load data (first download the data file into the working directory)
	music_file_path = 'features.csv'
	original = np.array(pd.read_csv(music_file_path, index_col=0, header=[0, 1, 2]))
	n_samples = int(original.shape[0]/5)
	n_indices = rng.choice(np.arange(original.shape[0]), n_samples, replace=False)
	original = original[n_indices, :]
	original = StandardScaler().fit_transform(original).T
	N = original.shape[1]
	d = original.shape[0]

	print 'N = ', N
	print 'd = ', d

	E_XX_T = np.dot(original, original.T)/N
	U_star, Sig_star, VT_star = np.linalg.svd(E_XX_T)

	x = np.arange(1, len(Sig_star)+1)
	plt.plot(x, Sig_star)
	plt.grid(which='both')
	plt.ylim(-1, np.max(Sig_star)+1)
	plt.show()

	want = 50
	x = np.arange(1, want+1)
	plt.plot(x, Sig_star[1:want+1])
	plt.grid(which='both')
	plt.show()

	n_iter = 200
	#n_list = np.arange(1, 1000, 50)
	n_list = np.array([317, 440, 611, 849, 1179, 1638, 2276])
	# n_list = [50]
	k = 10
	c = 7
	n_vects = int(np.minimum(N, c*k))
	print '# of vectors: ', n_vects
	print 'n_list: ', n_list
	print

	top_subspace = np.dot(U_star[:, 1:k+1], U_star[:, 1:k+1].T)

	m = 50
	errs = []
	errs_weighted = []
	errs_weighted_ = []

	stds = []
	stds_weighted = []
	stds_weighted_ = []

	for n in n_list:
	    err = []
	    err_weighted = []
	    err_weighted_ = []
	    for i in range(n_iter):
	        UU_T, UU_T_weigthed, UU_T_weighted_ = weighted_cov_sum(original, n, m, N, d, rng, k, n_vects)
	        U, _, _ = np.linalg.svd(UU_T)
	        #print 'past SVD step'
	        U_weighted, _, _ = np.linalg.svd(UU_T_weigthed)
	        U_weighted_, _, _ = np.linalg.svd(UU_T_weighted_)
	        err += [np.linalg.norm(top_subspace - np.dot(U[:, 1:k+1], U[:, 1:k+1].T), ord='fro')]
	        err_weighted += [np.linalg.norm(top_subspace - np.dot(U_weighted[:, 1:k+1], U_weighted[:, 1:k+1].T), ord='fro')]
	        err_weighted_ += [np.linalg.norm(top_subspace - np.dot(U_weighted_[:, 1:k+1], U_weighted_[:, 1:k+1].T), ord='fro')]
	    errs += [np.mean(err)]
	    errs_weighted += [np.mean(err_weighted)]
	    errs_weighted_ += [np.mean(err_weighted_)]
	    stds += [np.std(err)]
	    stds_weighted += [np.std(err_weighted)]
	    stds_weighted_ += [np.std(err_weighted_)]
	    print 'n = ', n, ' done.'
	    print 'err = ', np.mean(err), ' std = ', np.std(err)
	    print 'err_weighted = ', np.mean(err_weighted), ' std_weighted = ', np.std(err_weighted)
	    print 'err_weighted_ = ', np.mean(err_weighted_), ' std_weighted_ = ', np.std(err_weighted_)
	    print

	plt.errorbar(n_list, errs, stds, c='r', lw=0.5, marker='*', label='unweighted')
	plt.errorbar(n_list, errs_weighted, stds_weighted, c='g', lw=0.5, marker='x', label='weighted k')
	plt.errorbar(n_list, errs_weighted_, stds_weighted_, c='b', lw=0.5, marker='+', label='weighted ck')
	plt.legend()

	plt.xlabel('n')
	plt.ylabel('frob error of top k vectors')

	plt.grid(which='both')
	plt.show()
