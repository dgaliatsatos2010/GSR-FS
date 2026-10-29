import os,sys
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('MKL_NUM_THREADS','1');os.environ.setdefault('NUMEXPR_NUM_THREADS','1')
from concurrent.futures import ProcessPoolExecutor
import numpy as np,pandas as pd
from gsrfs_revision_core import GSRSelector,null_independent
B=int(sys.argv[1]);REPS=int(sys.argv[2])
def one(rep):
 seed=930000+rep;X,_,_,_=null_independent(seed=seed,n=240,p=24,dist='gaussian')
 e=GSRSelector(n_features='auto',n_permutations=B,alpha=.05,max_pairs=5000,pair_sampler='balanced',support_engine='vectorized',residual_engine='auto',random_state=seed).fit(X)
 return dict(rep=rep,seed=seed,B=B,nonempty=int(e.n_features_selected_>0),selected_k=int(e.n_features_selected_),threshold=float(e.auto_threshold_),max_support=float(np.max(e.effective_support_)),mc_p_resolution=1/(B+1))
if __name__=='__main__':
 with ProcessPoolExecutor(max_workers=4) as ex: rows=list(ex.map(one,range(REPS),chunksize=2))
 d=pd.DataFrame(rows);d.to_csv(f'r2_B{B}_raw.csv',index=False);print(B,d.nonempty.mean(),d.nonempty.sum(),d.threshold.mean())
