from __future__ import annotations
import math, time, gzip, importlib.util
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import nnls
from scipy.io import arff
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, make_blobs, make_moons
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score

class GSRSelector(BaseEstimator, TransformerMixin):
    def __init__(self,n_features='auto',n_permutations=200,alpha=0.05,max_pairs=50000,pair_sampler='balanced',robust=True,clip_z=4.5,coverage_power=2.0,use_residual=True,residual_power=0.5,allow_empty=True,support_engine='vectorized',permutation_batch_size=16,residual_engine='auto',residual_max_iter=1000,residual_tol=1e-12,random_state=None,eps=1e-12,support_mode='calibrated'):
        self.n_features=n_features; self.n_permutations=n_permutations; self.alpha=alpha; self.max_pairs=max_pairs; self.pair_sampler=pair_sampler; self.robust=robust; self.clip_z=clip_z; self.coverage_power=coverage_power; self.use_residual=use_residual; self.residual_power=residual_power; self.allow_empty=allow_empty; self.support_engine=support_engine; self.permutation_batch_size=permutation_batch_size; self.residual_engine=residual_engine; self.residual_max_iter=residual_max_iter; self.residual_tol=residual_tol; self.random_state=random_state; self.eps=eps; self.support_mode=support_mode
    def _coerce_X_and_names(self,X,feature_names=None):
        inferred=[str(c) for c in X.columns] if hasattr(X,'columns') else None
        X_arr=np.asarray(X,dtype=float)
        if X_arr.ndim!=2 or X_arr.shape[0]<3 or X_arr.shape[1]<2: raise ValueError('bad X')
        if not np.isfinite(X_arr).all(): raise ValueError('X contains NaN or infinite')
        names=[str(x) for x in feature_names] if feature_names is not None else (inferred if inferred is not None else [f'x{i}' for i in range(X_arr.shape[1])])
        return X_arr,np.asarray(names,dtype=object)
    def _scale_fit(self,X):
        eps=float(self.eps)
        if bool(self.robust):
            center=np.median(X,axis=0); mad=np.median(np.abs(X-center),axis=0); scale=1.4826*mad
            q25,q75=np.percentile(X,[25,75],axis=0); iqr=(q75-q25)/1.349; sd=X.std(axis=0,ddof=0)
            scale=np.where(scale>eps,scale,iqr); scale=np.where(scale>eps,scale,sd); scale=np.where(scale>eps,scale,1.0)
            Z=(X-center)/scale
            if self.clip_z is not None: Z=np.clip(Z,-float(self.clip_z),float(self.clip_z))
        else:
            center=X.mean(axis=0); scale=X.std(axis=0,ddof=0); scale=np.where(scale>eps,scale,1.0); Z=(X-center)/scale
        self.location_=center; self.scale_=scale; return Z
    def _sample_pairs_random(self,n,rng,target):
        pairs={}; batch=8192
        while len(pairs)<target:
            a=rng.integers(0,n,size=batch); b=rng.integers(0,n,size=batch); mask=a!=b
            lo=np.minimum(a[mask],b[mask]); hi=np.maximum(a[mask],b[mask])
            for pair in zip(lo.tolist(),hi.tolist()):
                if pair not in pairs:
                    pairs[pair]=None
                    if len(pairs)>=target: break
        arr=np.asarray(list(pairs.keys()),dtype=int); return arr[:,0],arr[:,1]
    def _sample_pairs_balanced(self,n,rng,target):
        real=rng.permutation(np.arange(n,dtype=int)).tolist(); dummy=-1; players=real+[dummy] if n%2 else real
        m=len(players); fixed=players[0]; rotating=players[1:]; oi=[]; oj=[]
        for _ in range(m-1):
            current=[fixed]+rotating; half=m//2
            for q in range(half):
                a=current[q]; b=current[m-1-q]
                if a==dummy or b==dummy: continue
                if a<b: oi.append(a); oj.append(b)
                else: oi.append(b); oj.append(a)
                if len(oi)>=target: return np.asarray(oi,dtype=int),np.asarray(oj,dtype=int)
            rotating=[rotating[-1]]+rotating[:-1]
        return np.asarray(oi,dtype=int),np.asarray(oj,dtype=int)
    def _sample_pairs(self,n,rng):
        total=n*(n-1)//2
        if total<=int(self.max_pairs): return np.triu_indices(n,1)
        target=int(self.max_pairs)
        return self._sample_pairs_balanced(n,rng,target) if self.pair_sampler=='balanced' else self._sample_pairs_random(n,rng,target)
    @staticmethod
    def _geometry_profiles(Xs,pair_i,pair_j):
        d=Xs[pair_i,:]-Xs[pair_j,:]; return d*d
    def _coverage_from_pair_mass(self,pair_mass,pair_i,pair_j,n):
        sm=np.bincount(pair_i,weights=pair_mass,minlength=n)+np.bincount(pair_j,weights=pair_mass,minlength=n)
        sm/=sm.sum()+float(self.eps); ent=-np.sum(sm*np.log(sm+float(self.eps))); return float(np.exp(ent)/n)
    @staticmethod
    def _pair_incidence(pair_i,pair_j,n):
        m=len(pair_i); rows=np.repeat(np.arange(m),2); cols=np.column_stack([pair_i,pair_j]).ravel(); data=np.ones(2*m)
        return sparse.csr_matrix((data,(rows,cols)),shape=(m,n))
    def _make_permutation_indices(self,n,rng):
        B=int(self.n_permutations); dtype=np.int32 if n<np.iinfo(np.int32).max else np.int64
        out=np.empty((B,n),dtype=dtype); base=np.arange(n,dtype=dtype)
        for b in range(B): out[b]=rng.permutation(base)
        return out
    def _support_scores_vectorized(self,Xs,G,pair_i,pair_j,permutation_indices):
        n,p=Xs.shape; B=int(self.n_permutations); eps=float(self.eps); cp=float(self.coverage_power); bs=int(self.permutation_batch_size)
        incidence=self._pair_incidence(pair_i,pair_j,n)
        P=G/(G.sum(axis=0,keepdims=True)+eps); total=P.sum(axis=1)
        robust_z=np.zeros(p); affinity=np.zeros(p); null_med=np.zeros(p); null_scale=np.zeros(p); coverage=np.zeros(p); null_eff=np.zeros((p,B))
        for j in range(p):
            rest=(total-P[:,j])/max(p-1,1); rest=rest/(rest.sum()+eps); pj=P[:,j]
            obs=float(np.sqrt(np.maximum(pj,0)*np.maximum(rest,0)).sum()); obs_cov=self._coverage_from_pair_mass(G[:,j],pair_i,pair_j,n)
            na=np.empty(B); nc=np.empty(B); xj=Xs[:,j]
            for start in range(0,B,bs):
                stop=min(B,start+bs); idx=permutation_indices[start:stop]; XP=xj[idx]; GP=(XP[:,pair_i]-XP[:,pair_j])**2
                PP=GP/(GP.sum(axis=1,keepdims=True)+eps); na[start:stop]=np.sqrt(np.maximum(PP,0)*np.maximum(rest[None,:],0)).sum(axis=1)
                sm=incidence.T.dot(GP.T).T; sm/=sm.sum(axis=1,keepdims=True)+eps; ent=-np.sum(sm*np.log(sm+eps),axis=1); nc[start:stop]=np.exp(ent)/n
            med=float(np.median(na)); mad=float(np.median(np.abs(na-med))); rs=1.4826*mad
            if rs<=eps: rs=float(np.std(na,ddof=1)) if B>1 else 0.0
            rs=max(rs,eps); zobs=max(0.0,(obs-med)/rs); znull=np.maximum(0.0,(na-med)/rs)
            affinity[j]=obs; null_med[j]=med; null_scale[j]=rs; robust_z[j]=zobs; coverage[j]=obs_cov; null_eff[j,:]=znull*np.power(nc,cp)
        eff=robust_z*np.power(coverage,cp)
        return robust_z,coverage,eff,affinity,null_med,null_scale,null_eff
    def _residuals_nnls(self,H,selected,remaining):
        if not selected: return np.ones(len(remaining))
        A=H[:,selected]; out=np.empty(len(remaining)); eps=float(self.eps)
        for q,j in enumerate(remaining):
            _,rn=nnls(A,H[:,j]); denom=float(H[:,j]@H[:,j])+eps; out[q]=np.clip((rn**2)/denom,0,1)
        return out
    def _residuals_batched_cd(self,H,selected,remaining):
        if not selected: return np.ones(len(remaining))
        S=np.asarray(selected,dtype=int); R=np.asarray(remaining,dtype=int); A=H[:,S]; Gss=A.T@A; C=A.T@H[:,R]; k,r=C.shape; coef=np.zeros((k,r)); diag=np.maximum(np.diag(Gss),float(self.eps))
        for _ in range(int(self.residual_max_iter)):
            mc=0.0
            for i in range(k):
                cross=Gss[i,:]@coef-Gss[i,i]*coef[i,:]; new=np.maximum(0.0,(C[i,:]-cross)/diag[i])
                if r: mc=max(mc,float(np.max(np.abs(new-coef[i,:]))))
                coef[i,:]=new
            if mc<=float(self.residual_tol): break
        GC=Gss@coef; tn=np.sum(H[:,R]*H[:,R],axis=0); res2=tn-2*np.sum(coef*C,axis=0)+np.sum(coef*GC,axis=0); res2=np.maximum(res2,0)
        return np.clip(res2/(tn+float(self.eps)),0,1)
    def _resolved_residual_engine(self,H):
        if self.residual_engine in {'nnls','batched_cd'}: return self.residual_engine
        if self.residual_engine=='auto': return 'batched_cd' if (H.shape[1]>=50 or H.shape[0]>=15000) else 'nnls'
        raise ValueError
    def _candidate_residuals(self,H,selected,remaining):
        if not bool(self.use_residual): return np.ones(len(remaining))
        return self._residuals_nnls(H,selected,remaining) if self._resolved_residual_engine(H)=='nnls' else self._residuals_batched_cd(H,selected,remaining)
    def fit(self,X,y=None,feature_names=None):
        X,names=self._coerce_X_and_names(X,feature_names); n,p=X.shape; self.n_features_in_=p; self.feature_names_in_=names
        seed_seq=np.random.SeedSequence(self.random_state); pair_seed,perm_seed=seed_seq.spawn(2); pair_rng=np.random.default_rng(pair_seed); perm_rng=np.random.default_rng(perm_seed)
        Xs=self._scale_fit(X); pi,pj=self._sample_pairs(n,pair_rng); G=self._geometry_profiles(Xs,pi,pj); perms=self._make_permutation_indices(n,perm_rng)
        support,cov,eff,aff,nmed,nscale,null_eff=self._support_scores_vectorized(Xs,G,pi,pj,perms)
        # Experimental ablation support representation; default reproduces frozen objective.
        if self.support_mode=='calibrated': base=eff; null_for_threshold=null_eff
        elif self.support_mode=='calibrated_no_coverage':
            base=support.copy(); null_for_threshold=np.zeros_like(null_eff)
            # recompute null z only from null_eff impossible after coverage multiplication; only fixed-k use supported.
        elif self.support_mode=='raw_affinity': base=aff.copy(); null_for_threshold=np.zeros_like(null_eff)
        elif self.support_mode=='raw_affinity_coverage': base=aff*np.power(cov,float(self.coverage_power)); null_for_threshold=np.zeros_like(null_eff)
        else: raise ValueError('unknown support_mode')
        if self.support_mode=='calibrated':
            max_null=np.max(null_for_threshold,axis=0)
            try: thr=float(np.quantile(max_null,1-float(self.alpha),method='higher'))
            except TypeError: thr=float(np.quantile(max_null,1-float(self.alpha),interpolation='higher'))
            adj=np.asarray([(1+np.sum(max_null>=s))/(len(max_null)+1) for s in base])
        else:
            max_null=np.full(int(self.n_permutations),np.nan); thr=-np.inf; adj=np.full(p,np.nan)
        H=G/(np.linalg.norm(G,axis=0,keepdims=True)+float(self.eps)); selected=[]; scores_out=np.zeros(p); res_at=np.full(p,np.nan); path=[]
        max_steps=p if self.n_features=='auto' else int(self.n_features)
        for step in range(max_steps):
            rem=[j for j in range(p) if j not in selected]
            if not rem: break
            residuals=self._candidate_residuals(H,selected,rem); rf=np.ones_like(residuals) if not self.use_residual else np.power(residuals,float(self.residual_power)); scores=base[np.asarray(rem)]*rf
            q=int(np.argmax(scores)); bj=int(rem[q]); bs=float(scores[q]); br=float(residuals[q])
            if self.n_features=='auto' and bs<thr: break
            selected.append(bj); scores_out[bj]=bs; res_at[bj]=br; path.append((step+1,bj,bs,br))
        if self.n_features=='auto' and not selected and not bool(self.allow_empty):
            bj=int(np.argmax(base)); selected=[bj]; scores_out[bj]=base[bj]; res_at[bj]=1.0
        self.support_scores_=support; self.sample_coverage_=cov; self.effective_support_=eff; self.affinity_=aff; self.null_affinity_median_=nmed; self.null_affinity_scale_=nscale; self.null_effective_support_=null_eff; self.max_null_distribution_=max_null; self.auto_threshold_=thr; self.adjusted_p_values_=adj; self.residual_novelty_=res_at; self.feature_scores_=scores_out; self.selected_indices_=np.asarray(selected,dtype=int); self.n_features_selected_=len(selected); self.selection_path_=path; self.n_pairs_used_=len(pi); self._H_=H
        self.support_=np.zeros(p,dtype=bool); self.support_[self.selected_indices_]=True
        rem=[j for j in np.argsort(-base) if j not in selected]; ordered=selected+rem; self.ranking_=np.empty(p,dtype=int)
        for rank,j in enumerate(ordered,1): self.ranking_[j]=rank
        return self
    def transform(self,X): return np.asarray(X)[:,self.support_]
    def get_support(self,indices=False): return self.selected_indices_.copy() if indices else self.support_.copy()

def _residual_fraction(target,basis,eps=1e-12):
    target=np.asarray(target,float); basis=np.asarray(basis,float); denom=float(target@target)+eps
    if basis.ndim!=2 or basis.shape[1]==0: return 1.0
    _,rn=nnls(basis,target); return float(np.clip((rn*rn)/denom,0,1))
def contextual_geometry_substitution(H,selected,representative,candidate,eps=1e-12):
    selected=[int(x) for x in selected]; r=int(representative); j=int(candidate)
    if r==j:return 1.0
    rc=_residual_fraction(H[:,j],H[:,selected],eps)
    replacement=[x for x in selected if x!=r]+[j]; rr=_residual_fraction(H[:,r],H[:,replacement],eps)
    return float(np.clip(1-max(rc,rr),0,1))
class GSRSubstitutionSelector(GSRSelector):
    def __init__(self,*args,substitution_threshold=0.85,**kwargs): super().__init__(*args,**kwargs); self.substitution_threshold=substitution_threshold
    def fit(self,X,y=None,feature_names=None):
        Xarr,names=self._coerce_X_and_names(X,feature_names); super().fit(Xarr,y=y,feature_names=names); selected=self.selected_indices_.tolist(); H=self._H_; p=Xarr.shape[1]
        sim=np.full((len(selected),p),np.nan); groups={int(r):[int(r)] for r in selected}; assigned=np.zeros(p)
        for a,r in enumerate(selected):
            for j in range(p):
                if j in selected and j!=r: continue
                sim[a,j]=contextual_geometry_substitution(H,selected,r,j,float(self.eps))
        for j in range(p):
            if j in selected or len(selected)==0: continue
            vals=sim[:,j]
            if np.isfinite(vals).any():
                a=int(np.nanargmax(vals)); val=float(vals[a])
                if val>=float(self.substitution_threshold): groups[int(selected[a])].append(int(j)); assigned[j]=val
        self.substitution_similarity_=sim; self.substitution_groups_={r:sorted(v) for r,v in groups.items()}; self.substitution_threshold_=float(self.substitution_threshold); self.substitution_assigned_similarity_=assigned
        return self
    def get_substitution_groups(self): return {int(k):list(v) for k,v in self.substitution_groups_.items()}

# synthetic generators from v0.16.2
def redundant_blocks(seed=0,n=240,noise_features=20):
    rng=np.random.default_rng(seed); y=np.repeat([0,1],n//2); y=np.r_[y,1] if y.size<n else y
    z1=rng.normal(np.where(y==0,-2.,2.),.55); z2=rng.normal(np.where(y==0,-1.4,1.4),.65)
    X=np.column_stack([z1,z2,z1+rng.normal(0,.05,n),z2+rng.normal(0,.05,n),rng.normal(size=(n,noise_features))]); return X,y,[[0,2],[1,3]],'redundant_blocks'
def nonlinear_moons(seed=0,n=240,noise_features=20):
    X2,y=make_moons(n_samples=n,noise=.08,random_state=seed); rng=np.random.default_rng(seed+1000); f0=X2[:,0]; f1=X2[:,1]
    X=np.column_stack([f0,f1,np.tanh(f0)+rng.normal(0,.02,n),np.sign(f1)*np.sqrt(np.abs(f1)+1e-6)+rng.normal(0,.02,n),rng.normal(size=(n,noise_features))]); return X,y,[[0,2],[1,3]],'nonlinear_moons'
def three_factor_blobs(seed=0,n=300,noise_features=30):
    Z,y=make_blobs(n_samples=n,centers=[[-3,0,2],[0,3,-2],[3,-2,0]],cluster_std=.7,n_features=3,random_state=seed); rng=np.random.default_rng(seed+2000)
    X=np.column_stack([Z[:,0],Z[:,1],Z[:,2],Z[:,0]+rng.normal(0,.08,n),Z[:,1]+rng.normal(0,.08,n),Z[:,2]+rng.normal(0,.08,n),rng.normal(size=(n,noise_features))]); return X,y,[[0,3],[1,4],[2,5]],'three_factor_blobs'
def outlier_contamination(seed=0,n=240,noise_features=20,outlier_fraction=.08):
    X,y,g,_=redundant_blocks(seed,n,noise_features); rng=np.random.default_rng(seed+3000); m=max(1,int(round(outlier_fraction*n))); idx=rng.choice(n,size=m,replace=False); X[idx,4:]+=rng.normal(0,12,size=(m,noise_features)); return X,y,g,'outlier_contamination'
def null_independent(seed=0,n=240,p=24,dist='gaussian',rho=0.0):
    rng=np.random.default_rng(seed)
    if dist=='gaussian': X=rng.normal(size=(n,p))
    elif dist=='t5': X=rng.standard_t(5,size=(n,p))
    elif dist=='correlated':
        cov=rho**np.abs(np.subtract.outer(np.arange(p),np.arange(p))); X=rng.multivariate_normal(np.zeros(p),cov,size=n)
    else: raise ValueError
    y=rng.integers(0,2,size=n); return X,y,[],'null_'+dist

def group_coverage(selected,groups):
    s=set(map(int,selected)); return float(np.mean([bool(s.intersection(g)) for g in groups])) if groups else np.nan
def relevant_precision(selected,groups):
    selected=list(map(int,selected)); rel={j for g in groups for j in g}; return float(np.mean([j in rel for j in selected])) if selected else 0.0
def redundancy_rate(selected,groups):
    s=set(map(int,selected)); extra=0
    for g in groups: extra+=max(0,len(s.intersection(g))-1)
    return float(extra/max(len(selected),1))

@dataclass(frozen=True)
class RealDataset:
    name:str; X:object; y:np.ndarray; feature_names:tuple[str,...]=(); groups:np.ndarray|None=None; source:str=''; panel_status:str=''

def downstream_k(p):
    p=int(p); return p if p<=2 else min(p-1,max(2,int(math.ceil(math.sqrt(p)))))
def _factorize(s):
    vals=pd.Series(s).reset_index(drop=True); codes,_=pd.factorize(vals.astype(str),sort=True); return codes.astype(int)
def _pkg_root(pkg): return Path(importlib.util.find_spec(pkg).origin).resolve().parent

def load_offline_real_panel(include_digits=True):
    out=[]
    for name,loader in [('iris',load_iris),('wine',load_wine),('breast_cancer',load_breast_cancer)]+([('digits',load_digits)] if include_digits else []):
        b=loader(); names=getattr(b,'feature_names',[f'x{i}' for i in range(b.data.shape[1])]); out.append(RealDataset(name,np.asarray(b.data,float),np.asarray(b.target),tuple(map(str,names)),None,'scikit-learn bundled','development_reused'))
    import statsmodels.datasets as smd
    anes=smd.anes96.load_pandas(); out.append(RealDataset('anes96',np.asarray(anes.exog,float),np.asarray(anes.endog,int),tuple(map(str,anes.exog_name)),None,'statsmodels.datasets.anes96','offline_extension'))
    mode=smd.modechoice.load_pandas(); out.append(RealDataset('modechoice',np.asarray(mode.exog,float),np.asarray(mode.endog,int),tuple(map(str,mode.exog_name)),np.asarray(mode.data['individual'],int),'statsmodels.datasets.modechoice','offline_extension'))
    sp=smd.spector.load_pandas(); out.append(RealDataset('spector',np.asarray(sp.exog,float),np.asarray(sp.endog,int),tuple(map(str,sp.exog_name)),None,'statsmodels.datasets.spector','offline_extension'))
    return out

def load_gate_extension_panel():
    sm=_pkg_root('statsmodels'); p9=_pkg_root('plotnine'); pl=_pkg_root('plotly'); out=[]
    f=sm/'gam/tests/results/autos.csv'; df=pd.read_csv(f); df=df.loc[df.symbol.notna()].reset_index(drop=True); out.append(RealDataset('autos_symbol',df.drop(columns=['symbol']),_factorize(df.symbol),(),None,str(f),'gate_extension_v014'))
    f=sm/'genmod/tests/results/stata_lbw_glm.csv'; df=pd.read_csv(f); df=df.loc[df.low.notna()].reset_index(drop=True); out.append(RealDataset('birthwt_low',df.drop(columns=[c for c in ['low','bwt','id'] if c in df]),_factorize(df.low),(),None,str(f),'gate_extension_v014'))
    f=sm/'treatment/tests/results/cataneo2.csv'; df=pd.read_csv(f); 
    if len(df)>600:
        rng=np.random.default_rng(1414); df=df.iloc[np.sort(rng.choice(len(df),600,replace=False))].reset_index(drop=True)
    y=_factorize(df['mbsmoke_']); drop={'mbsmoke_','mbsmoke','msmoke','bweight','lbweight','fbaby_','mmarried_','prenatal1_'}; out.append(RealDataset('cattaneo_smoking',df.drop(columns=[c for c in drop if c in df]),y,(),None,str(f),'gate_extension_v014'))
    f=sm/'genmod/tests/results/epil.csv'; df=pd.read_csv(f); out.append(RealDataset('epil_treatment',df.drop(columns=[c for c in ['trt','subject','Unnamed: 0'] if c in df]),_factorize(df.trt),(),df.subject.to_numpy(copy=True),str(f),'gate_extension_v014'))
    f=sm/'stats/tests/results/framing.csv'; df=pd.read_csv(f); out.append(RealDataset('framing_congress_message',df[['emo','treat','age','educ','gender','income']],_factorize(df.cong_mesg),(),None,str(f),'gate_extension_v014'))
    f=pl/'package_data/datasets/gapminder.csv.gz'; df=pd.read_csv(f); countries=np.array(sorted(df.country.dropna().unique().tolist()),object); rng=np.random.default_rng(1414); chosen=set(rng.choice(countries,size=min(60,len(countries)),replace=False).tolist()); df=df.loc[df.country.isin(chosen)].sort_values(['country','year']).reset_index(drop=True); out.append(RealDataset('gapminder_year',df[['continent','lifeExp','pop','gdpPercap']],_factorize(df.year),(),df.country.astype(str).to_numpy(),str(f),'gate_extension_v014'))
    f=p9/'data/midwest.csv'; df=pd.read_csv(f); out.append(RealDataset('midwest_state',df.drop(columns=[c for c in {'state','PID','county','category','inmetro'} if c in df]),_factorize(df.state),(),None,str(f),'gate_extension_v014'))
    return out

def _factorize_target(s):
    s=pd.Series(s).reset_index(drop=True)
    if pd.api.types.is_bool_dtype(s) or pd.api.types.is_numeric_dtype(s):
        uniq=sorted(s.dropna().unique().tolist()); mp={v:i for i,v in enumerate(uniq)}; return s.map(mp).to_numpy(dtype=int)
    return pd.factorize(s.astype(str),sort=True)[0].astype(int)
def _uniform_cap(df,cap,seed=1401):
    if cap is None or len(df)<=cap:return df.reset_index(drop=True)
    rng=np.random.default_rng(seed); return df.iloc[np.sort(rng.choice(len(df),cap,replace=False))].reset_index(drop=True)
def _finalize(name,df,target,drop=(),cap=None,derive=None,status='auxiliary_v014'):
    df=df.copy()
    if derive is None: df=df.loc[df[target].notna()].reset_index(drop=True)
    else: df=df.loc[df[derive['base']].notna()].reset_index(drop=True)
    df=_uniform_cap(df,cap)
    ys=df[target] if derive is None else (df[derive['base']].astype(float)>float(derive['value'])).astype(int)
    y=_factorize_target(ys); drops=set(drop); drops.add(target if derive is None else derive['base']); X=df.drop(columns=[c for c in drops if c in df])
    return RealDataset(name,X,y,(),None,'bundled public data',status)
def load_auxiliary_panel():
    sm=_pkg_root('statsmodels'); p9=_pkg_root('plotnine'); gr=_pkg_root('gradio'); out=[]
    out.append(_finalize('ccard_ownrent',pd.read_csv(sm/'datasets/ccard/ccard.csv'),'OWNRENT'))
    out.append(_finalize('heart_censor',pd.read_csv(sm/'datasets/heart/heart.csv'),'censors'))
    out.append(_finalize('fair_affair_any',pd.read_csv(sm/'datasets/fair/fair.csv'),'affair_any',cap=900,derive={'base':'affairs','value':0}))
    out.append(_finalize('bmt_status',pd.read_csv(sm/'duration/tests/results/bmt.csv'),'Status'))
    out.append(_finalize('ologit_apply',pd.read_csv(sm/'miscmodels/tests/results/ologit_ucla.csv'),'apply'))
    out.append(_finalize('constrict_diagnosis',pd.read_csv(sm/'stats/tests/results/binary_constrict.csv'),'diagnosis',drop=('Case',)))
    out.append(_finalize('ships_type',pd.read_csv(sm/'discrete/tests/results/ships.csv'),'ship'))
    out.append(_finalize('mtcars_transmission',pd.read_csv(p9/'data/mtcars.csv'),'am',drop=('name',)))
    out.append(_finalize('penguins_species',pd.read_csv(p9/'data/penguins.csv'),'species'))
    out.append(_finalize('mpg_class',pd.read_csv(p9/'data/mpg.csv'),'class'))
    out.append(_finalize('diamonds_cut',pd.read_csv(p9/'data/diamonds.csv'),'cut',cap=900))
    out.append(_finalize('msleep_vore',pd.read_csv(p9/'data/msleep.csv'),'vore',drop=('name',)))
    out.append(_finalize('titanic_survived',pd.read_csv(gr/'media_assets/data/titanic.csv'),'Survived',drop=('PassengerId',)))
    return out

def make_tabular_preprocessor(X):
    X=pd.DataFrame(X); num=list(X.select_dtypes(include=[np.number,'bool']).columns); cat=[c for c in X.columns if c not in num]; trs=[]
    if num: trs.append(('num',Pipeline([('impute',SimpleImputer(strategy='median'))]),num))
    if cat: trs.append(('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='infrequent_if_exist',min_frequency=.02,sparse_output=False))]),cat))
    return ColumnTransformer(trs,remainder='drop',sparse_threshold=0.0)

def load_openml_fixtures():
    root=_pkg_root('sklearn')/'datasets/tests/data/openml'; specs=[(3,'openml3_kr_vs_kp','class',[],900),(40675,'openml40675_glass2','class',[],None),(62,'openml62_zoo','type',['animal'],None)]; out=[]
    for did,name,target,ignore,cap in specs:
        folder=root/f'id_{did}'; f=next(folder.glob('data-v1-dl-*.arff.gz'))
        with gzip.open(f,'rt') as g: data,_=arff.loadarff(g)
        df=pd.DataFrame(data)
        for c in df.columns:
            if df[c].dtype==object: df[c]=df[c].map(lambda v:v.decode('utf-8') if isinstance(v,(bytes,bytearray)) else v)
        if cap is not None and len(df)>cap:
            rng=np.random.default_rng(1401); df=df.iloc[np.sort(rng.choice(len(df),size=cap,replace=False))].reset_index(drop=True)
        y=pd.factorize(df[target].astype(str),sort=True)[0].astype(int); X=df.drop(columns=[c for c in [target]+ignore if c in df]); out.append(RealDataset(name,X,y,(),None,str(f),'openml_fixture_extension_v014'))
    return out

def all_30_datasets(): return load_offline_real_panel(True)+load_gate_extension_panel()+load_auxiliary_panel()+load_openml_fixtures()

def preprocess_fold(ds,tr,te):
    if isinstance(ds.X,pd.DataFrame):
        Xtr_raw=ds.X.iloc[tr].copy(); Xte_raw=ds.X.iloc[te].copy(); pre=make_tabular_preprocessor(Xtr_raw); Xtr=pre.fit_transform(Xtr_raw); Xte=pre.transform(Xte_raw); sc=StandardScaler(); return sc.fit_transform(Xtr),sc.transform(Xte)
    return np.asarray(ds.X,float)[tr],np.asarray(ds.X,float)[te]

def split_dataset(ds,seed=0,n_splits=3):
    if ds.groups is None: sp=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed); return list(sp.split(ds.X,ds.y)),'stratified'
    sp=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed); return list(sp.split(ds.X,ds.y,groups=ds.groups)),'stratified_group'

def classifier_obj(name,seed=0):
    if name=='logistic': return make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight='balanced'))
    if name=='linear_svm': return make_pipeline(StandardScaler(),LinearSVC(class_weight='balanced',random_state=seed,max_iter=10000))
    if name=='random_forest': return RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=seed,n_jobs=1)
    raise ValueError
