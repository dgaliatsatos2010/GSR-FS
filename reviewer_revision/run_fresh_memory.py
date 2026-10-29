import subprocess,concurrent.futures,pandas as pd,os
cfg=[]
for n in [100,250,500,1000,2000]: cfg.append(('n',n,n,50,10000,50))
for p in [20,50,100,200,400]: cfg.append(('p',p,500,p,10000,50))
for M in [1000,2500,5000,10000,20000,50000]: cfg.append(('pair_budget',M,500,50,M,50))
for B in [25,50,100,200]: cfg.append(('permutations',B,500,50,10000,B))
def one(c):
 axis,val,n,p,M,B=c
 env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 out=subprocess.check_output(['python','/mnt/data/scalability_worker_fresh.py','--n',str(n),'--p',str(p),'--M',str(M),'--B',str(B)],text=True,env=env).strip()
 sec,rss,pairs=out.split(',');return dict(axis=axis,value=val,n=n,p=p,max_pairs=M,n_permutations=B,fit_seconds=float(sec),peak_rss_mb=float(rss),pairs=int(pairs))
rows=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
 for r in ex.map(one,cfg): rows.append(r);print(r,flush=True)
df=pd.DataFrame(rows);df.to_csv('/mnt/data/scalability_fresh_memory.csv',index=False);print(df.to_string(index=False))
