import sys,time
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import *
rows=[]
for ds in all_30_datasets():
    splits,kind=split_dataset(ds,0,3)
    for fold,(tr,te) in enumerate(splits):
        Xtr,Xte=preprocess_fold(ds,tr,te); ytr=ds.y[tr]; yte=ds.y[te]
        k=downstream_k(Xtr.shape[1])
        seed=(141400+fold) if ds.panel_status=='gate_extension_v014' else (fold if ds.panel_status in {'auxiliary_v014','openml_fixture_extension_v014'} else fold)
        sel=GSRSelector(n_features=k,n_permutations=12,max_pairs=8000,pair_sampler='balanced',random_state=seed,support_engine='vectorized',residual_engine='auto').fit(Xtr)
        idx=sel.get_support(indices=True)
        clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
        clf.fit(Xtr[:,idx],ytr); pred=clf.predict(Xte[:,idx])
        rows.append(dict(dataset=ds.name,fold=fold,k=k,selected=','.join(map(str,idx)),balanced_accuracy=balanced_accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro',zero_division=0),accuracy=accuracy_score(yte,pred)))
    print(ds.name, 'done')
pd.DataFrame(rows).to_csv('/mnt/data/gsr_fixed_validation_raw.csv',index=False)
s=pd.DataFrame(rows).groupby('dataset')[['balanced_accuracy','macro_f1','accuracy']].mean().reset_index(); s.to_csv('/mnt/data/gsr_fixed_validation_summary.csv',index=False); print(s.to_string(index=False))
