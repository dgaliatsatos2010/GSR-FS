import sys,time,os
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import *
out='/mnt/data/auto_external_B200_M8000_raw.csv'
rows=[]
if os.path.exists(out): rows=pd.read_csv(out).to_dict('records')
done={(r['dataset'],int(r['fold'])) for r in rows}
for di,ds in enumerate(all_30_datasets()):
    splits,kind=split_dataset(ds,0,3)
    for fold,(tr,te) in enumerate(splits):
        if (ds.name,fold) in done: continue
        Xtr,Xte=preprocess_fold(ds,tr,te); ytr=ds.y[tr]; yte=ds.y[te]
        seed=(141400+fold) if ds.panel_status=='gate_extension_v014' else fold
        t0=time.perf_counter(); sel=GSRSelector(n_features='auto',n_permutations=200,alpha=.05,max_pairs=8000,pair_sampler='balanced',robust=True,coverage_power=2.0,use_residual=True,residual_power=.5,allow_empty=True,support_engine='vectorized',residual_engine='auto',random_state=seed).fit(Xtr); fit=time.perf_counter()-t0; idx=sel.get_support(indices=True)
        if len(idx)==0:
            clf=DummyClassifier(strategy='prior'); clf.fit(np.zeros((len(ytr),1)),ytr); pred=clf.predict(np.zeros((len(yte),1)))
        else:
            clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced')); clf.fit(Xtr[:,idx],ytr); pred=clf.predict(Xte[:,idx])
        rows.append(dict(dataset=ds.name,panel_status=ds.panel_status,fold=fold,split_kind=kind,n_samples=len(ds.y),n_features=Xtr.shape[1],selected_k=len(idx),selected_frac=len(idx)/Xtr.shape[1],empty=int(len(idx)==0),auto_threshold=sel.auto_threshold_,balanced_accuracy=balanced_accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro',zero_division=0),accuracy=accuracy_score(yte,pred),selector_fit_seconds=fit,selected=','.join(map(str,idx))))
        pd.DataFrame(rows).to_csv(out,index=False)
        print(di+1,ds.name,fold,'k',len(idx),'time',round(fit,3),flush=True)
df=pd.DataFrame(rows)
s=df.groupby('dataset').agg(n_features=('n_features','mean'),mean_auto_k=('selected_k','mean'),sd_auto_k=('selected_k','std'),min_auto_k=('selected_k','min'),max_auto_k=('selected_k','max'),empty_rate=('empty','mean'),selected_frac=('selected_frac','mean'),balanced_accuracy=('balanced_accuracy','mean'),macro_f1=('macro_f1','mean'),accuracy=('accuracy','mean'),fit_seconds=('selector_fit_seconds','mean')).reset_index(); s.to_csv('/mnt/data/auto_external_B200_M8000_summary.csv',index=False)
print('\nOVERALL'); print(df[['selected_k','selected_frac','empty','balanced_accuracy','macro_f1','accuracy','selector_fit_seconds']].mean().to_string()); print(s.to_string(index=False))
