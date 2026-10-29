import sys
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd
sys.path.insert(0,'/mnt/data')
from gsrfs_revision_core import *
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score,f1_score,accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

def work(di):
    datasets=all_30_datasets(); ds=datasets[di]
    fixed=pd.read_csv('/mnt/data/gsr_fixed_validation_raw.csv'); auto=pd.read_csv('/mnt/data/auto_external_B200_M8000_raw.csv')
    selmap={}
    for mode,df in [('fixed',fixed),('auto',auto)]:
        q=df[df.dataset==ds.name]
        for _,r in q.iterrows(): selmap[(mode,int(r.fold))]=[int(x) for x in str(r.selected).split(',') if x and x!='nan']
    rows=[]; splits,kind=split_dataset(ds,0,3)
    for fold,(tr,te) in enumerate(splits):
        Xtr,Xte=preprocess_fold(ds,tr,te); ytr=ds.y[tr]; yte=ds.y[te]
        for mode in ['fixed','auto']:
            idx=np.asarray(selmap[(mode,fold)],dtype=int)
            for clf_name in ['logistic','linear_svm','random_forest']:
                if idx.size==0:
                    clf=DummyClassifier(strategy='prior'); clf.fit(np.zeros((len(ytr),1)),ytr); pred=clf.predict(np.zeros((len(yte),1)))
                else:
                    if clf_name=='logistic': clf=make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
                    elif clf_name=='linear_svm': clf=make_pipeline(StandardScaler(),LinearSVC(class_weight='balanced',max_iter=10000,dual='auto',random_state=fold))
                    else: clf=RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=100+fold,n_jobs=1)
                    clf.fit(Xtr[:,idx],ytr); pred=clf.predict(Xte[:,idx])
                rows.append(dict(dataset=ds.name,fold=fold,mode=mode,classifier=clf_name,k=len(idx),balanced_accuracy=balanced_accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro',zero_division=0),accuracy=accuracy_score(yte,pred)))
    return rows
rows=[]
with ProcessPoolExecutor(max_workers=5) as ex:
    fs={ex.submit(work,i):i for i in range(30)}
    for f in as_completed(fs): rows.extend(f.result()); print('done',fs[f],flush=True)
raw=pd.DataFrame(rows).sort_values(['dataset','fold','mode','classifier']); raw.to_csv('/mnt/data/classifier_sensitivity_raw.csv',index=False)
ds=raw.groupby(['dataset','mode','classifier'],as_index=False)[['balanced_accuracy','macro_f1','accuracy']].mean(); ds.to_csv('/mnt/data/classifier_sensitivity_dataset_summary.csv',index=False)
overall=ds.groupby(['mode','classifier'],as_index=False)[['balanced_accuracy','macro_f1','accuracy']].mean(); overall.to_csv('/mnt/data/classifier_sensitivity_overall.csv',index=False)
from scipy.stats import wilcoxon
post=[]
for clf in sorted(ds.classifier.unique()):
 d=ds[ds.classifier==clf].pivot(index='dataset',columns='mode',values='balanced_accuracy').dropna(); diff=d.auto-d.fixed
 stat,p=wilcoxon(diff,zero_method='wilcox') if not np.allclose(diff,0) else (0,1)
 post.append(dict(classifier=clf,n=len(d),mean_auto=d.auto.mean(),mean_fixed=d.fixed.mean(),mean_diff=diff.mean(),median_diff=diff.median(),wilcoxon_p=p))
post=pd.DataFrame(post); post.to_csv('/mnt/data/classifier_sensitivity_auto_vs_fixed.csv',index=False)
print(overall.to_string(index=False)); print('\n',post.to_string(index=False))
