import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh
from sklearn.linear_model import Lasso
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors

class _TopK:
    def _finish(self,X,scores,k,higher=True):
        order=np.argsort(-scores if higher else scores); self.selected_indices_=np.asarray(order[:int(k)],int); return self

def _std(X):
    X=np.asarray(X,float);mu=X.mean(0);sd=X.std(0);sd=np.where(sd>1e-12,sd,1.);return (X-mu)/sd

def _graph(Z,n_neighbors=7):
    n=len(Z);k=min(max(int(n_neighbors)+1,2),n);dist,ind=NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z);d2=dist[:,1:]**2;nbr=ind[:,1:];pos=d2[d2>0];t=float(np.median(pos)) if pos.size else 1.;w=np.exp(-d2/max(t,1e-12));rows=np.repeat(np.arange(n),nbr.shape[1]);W=sparse.coo_matrix((w.ravel(),(rows,nbr.ravel())),shape=(n,n)).tocsr();return W.maximum(W.T)

class MaxVarianceSelector(_TopK):
    def __init__(self,n_features=5):self.n_features=n_features
    def fit(self,X,y=None):return self._finish(X,np.var(np.asarray(X,float),axis=0),self.n_features,True)

class LaplacianScoreSelector(_TopK):
    def __init__(self,n_features=5,n_neighbors=5):self.n_features=n_features;self.n_neighbors=n_neighbors
    def fit(self,X,y=None):
        X=np.asarray(X,float);Z=_std(X);W=_graph(Z,self.n_neighbors);d=np.asarray(W.sum(1)).ravel();L=sparse.diags(d)-W;ds=d.sum()+1e-12;s=np.empty(Z.shape[1])
        for j in range(Z.shape[1]):
            f=Z[:,j];f=f-(f@d)/ds;den=float(f@(d*f));s[j]=float(f@(L@f))/(den+1e-12)
        return self._finish(X,s,self.n_features,False)

class SPECSelector(_TopK):
    def __init__(self,n_features=5,gamma=1.0):self.n_features=n_features;self.gamma=gamma
    def fit(self,X,y=None):
        X=np.asarray(X,float);Z=X.copy();n,p=Z.shape;W=rbf_kernel(Z,gamma=float(self.gamma));d=W.sum(1);ds=np.maximum(d,1e-12);di=1/np.sqrt(ds);L=np.diag(d)-W;Lh=(di[:,None]*L)*di[None,:];vals,U=np.linalg.eigh(Lh);vals=vals[::-1];U=U[:,::-1];v=np.sqrt(ds);v/=np.linalg.norm(v)+1e-12;s=np.full(p,-np.inf)
        for j in range(p):
            fh=np.sqrt(ds)*Z[:,j];norm=np.linalg.norm(fh)
            if norm<=100*np.spacing(1):continue
            fh/=norm;a=(fh@U)**2;den=1-float((fh@v)**2);s[j]=float(np.sum(a[:-1]*vals[:-1])/(den+1e-12))
        return self._finish(X,s,self.n_features,True)

def estimate_clusters(X,n_neighbors=7,max_clusters=10):
    Z=_std(X);n=len(Z);W=_graph(Z,n_neighbors);d=np.asarray(W.sum(1)).ravel();di=1/np.sqrt(np.maximum(d,1e-12));L=sparse.eye(n)-sparse.diags(di)@W@sparse.diags(di);upper=min(max_clusters,n-2);nev=min(upper+1,n-1);vals=np.sort(np.real(eigsh(L,k=nev,which='SM',return_eigenvectors=False)));cand=np.arange(2,min(upper,len(vals)-1)+1);g=np.array([vals[k]-vals[k-1] for k in cand]);return int(cand[np.argmax(g)]) if len(cand) else 2

class MCFSSelector(_TopK):
    def __init__(self,n_features=5,n_clusters='eigengap',n_neighbors=5,alpha=.01):self.n_features=n_features;self.n_clusters=n_clusters;self.n_neighbors=n_neighbors;self.alpha=alpha
    def fit(self,X,y=None):
        X=np.asarray(X,float);Z=_std(X);n=len(Z);c=estimate_clusters(Z,self.n_neighbors) if self.n_clusters=='eigengap' else int(self.n_clusters);W=_graph(Z,self.n_neighbors);d=np.asarray(W.sum(1)).ravel();di=1/np.sqrt(np.maximum(d,1e-12));L=sparse.eye(n)-sparse.diags(di)@W@sparse.diags(di);nv=min(max(c+1,2),n-1);vals,vecs=eigsh(L,k=nv,which='SM');ord=np.argsort(vals);Y=vecs[:,ord[1:min(nv,c+1)]];co=[]
        for q in range(Y.shape[1]):
            m=Lasso(alpha=float(self.alpha),fit_intercept=True,max_iter=5000);m.fit(Z,Y[:,q]);co.append(np.abs(m.coef_))
        s=np.max(np.vstack(co),axis=0) if co else np.var(Z,axis=0);self.n_clusters_estimated_=c;return self._finish(X,s,self.n_features,True)
