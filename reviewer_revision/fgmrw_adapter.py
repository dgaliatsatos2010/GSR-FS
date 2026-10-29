import numpy as np
from scipy.spatial.distance import cdist

EPS=1e-12

def get_radius(gb):
    X=gb[:,:-1]; c=X.mean(0); return np.sqrt(((X-c)**2).sum(1)).max()
def split2(data):
    X=data[:,:-1]; D=cdist(X,X,'euclidean'); r,c=np.where(D==np.max(D)); r1=r[1]; c1=c[1]
    cl=np.where(D[:,r1] < D[:,c1],0,1); return [data[cl==0],data[cl==1]]
def density(gb):
    X=gb[:,:-1]; c=X.mean(0); s=np.sqrt(((X-c)**2).sum(1)).sum(); return s if s!=0 else len(X)
def division(lst,nmin):
    out=[]
    for gb in lst:
        if len(gb)>=nmin:
            b1,b2=split2(gb); dp=density(gb); d1=density(b1); d2=density(b2); w=len(b1)+len(b2)
            if (len(b1)/w*d1 + len(b2)/w*d2) < dp: out.extend([b1,b2])
            else: out.append(gb)
        else: out.append(gb)
    return out
def normalized(lst,rd):
    out=[]
    for gb in lst:
        if len(gb)<2: out.append(gb)
        else:
            b1,b2=split2(gb)
            if get_radius(gb)<=2*rd: out.append(gb)
            else: out.extend([b1,b2])
    return out
def getGranularBall2(data):
    n=data.shape[0]; idx=np.arange(n).reshape(n,1); lst=[np.hstack((data,idx))]
    rt=n**0.5
    if rt>64: nmin=64
    if rt>32: nmin=32
    if rt>16: nmin=16
    elif rt>8: nmin=8
    else: nmin=int(rt)
    while True:
        old=len(lst); lst=division(lst,nmin)
        if len(lst)==old: break
    radii=[get_radius(g) for g in lst if len(g)>=2]; rd=max(np.median(radii),np.mean(radii))
    while True:
        old=len(lst); lst=normalized(lst,rd)
        if len(lst)==old: break
    centers=np.vstack([g.mean(0) for g in lst])[:,:-1]
    return centers

def safe_divide(a,b,fill=0.):
    with np.errstate(divide='ignore',invalid='ignore'): r=np.divide(a,b)
    return np.where(np.isfinite(r),r,fill)
def l1_projection(x,s=1.):
    u=np.abs(x)
    if u.sum()<=s:return x
    theta=np.sort(u)[::-1]; cs=np.cumsum(theta); rho=np.max(np.where(theta*(np.arange(len(x))+1)>(cs-s))[0]); lam=(cs[rho]-s)/(rho+1)
    return np.sign(x)*np.maximum(u-lam,0.)
def gaussian_matrix(a,b,delta):
    diff=a[:,None]-b[None,:]; return np.exp(-(diff**2)/(2*delta**2+EPS))
def matching_matrix(a,b): return (a[:,None]==b[None,:]).astype(float)

def fgmrw_ufs(data,alpha=0.1,s=0.8,beta=0.2,max_iter=10000,tol=1e-4,decay_rate=0.1):
    # Author implementation interprets a [0,1]-ranged feature as numerical; min-max scale upstream.
    data=np.asarray(data,float); gb=getGranularBall2(data); n,m=data.shape
    Mc=[]
    for j in range(m):
        col=data[:,j]; cgb=gb[:,j]
        if np.min(col)==0 and np.max(col)==1: r=gaussian_matrix(col,cgb,alpha)
        else: r=matching_matrix(col,cgb)
        Mc.append(r)
    FSG=np.zeros((m,m),float)
    for i in range(m):
        dri=Mc[i]; one=1-dri
        for j in range(m):
            drj=Mc[j]; amin=np.min(np.maximum(one,drj),axis=1); amax=np.max(np.minimum(dri,drj),axis=1)
            FSG[i,j]=np.max(safe_divide(amin,amax,0.))
    deg=FSG.sum(1); P=safe_divide(FSG,deg[:,None]+1e-8,0.); d=1/(1+np.log(np.mean(deg)+1+EPS))
    pi0=np.ones(m)/m; pit=pi0.copy(); pip=pi0.copy(); visits=np.zeros(m)
    for _ in range(max_iter):
        mom=(1-beta)*pit+beta*pip; mom*=1/(1+decay_rate*visits); new=d*pi0+(1-d)*(mom@P)
        sm=new.sum()
        if sm>EPS: visits += new/sm
        new=l1_projection(new,s)
        if np.linalg.norm(new-pit,1)<tol: pit=new; break
        pip=pit.copy(); pit=new
    mn,mx=pit.min(),pit.max(); score=np.zeros(m) if mx-mn<EPS else (pit-mn)/(mx-mn)
    return np.argsort(-score), score

def fgmrw_ufs_vectorized(data,alpha=0.1,s=0.8,beta=0.2,max_iter=10000,tol=1e-4,decay_rate=0.1):
    data=np.asarray(data,float); gb=getGranularBall2(data); n,m=data.shape
    mats=[]
    for j in range(m):
        col=data[:,j]; cgb=gb[:,j]
        mats.append(gaussian_matrix(col,cgb,alpha) if (np.min(col)==0 and np.max(col)==1) else matching_matrix(col,cgb))
    M=np.stack(mats,axis=0)  # m,n,n_gb
    FSG=np.zeros((m,m),float)
    for i in range(m):
        dri=M[i]; one=1-dri
        lap_min=np.min(np.maximum(one[None,:,:],M),axis=2)  # m,n
        lap_max=np.max(np.minimum(dri[None,:,:],M),axis=2)  # m,n
        ratios=safe_divide(lap_min,lap_max,0.)
        FSG[i,:]=np.max(ratios,axis=1)
    deg=FSG.sum(1); P=safe_divide(FSG,deg[:,None]+1e-8,0.); d=1/(1+np.log(np.mean(deg)+1+EPS))
    pi0=np.ones(m)/m; pit=pi0.copy(); pip=pi0.copy(); visits=np.zeros(m)
    for _ in range(max_iter):
        mom=(1-beta)*pit+beta*pip; mom*=1/(1+decay_rate*visits); new=d*pi0+(1-d)*(mom@P)
        sm=new.sum()
        if sm>EPS: visits += new/sm
        new=l1_projection(new,s)
        if np.linalg.norm(new-pit,1)<tol: pit=new; break
        pip=pit.copy(); pit=new
    mn,mx=pit.min(),pit.max(); score=np.zeros(m) if mx-mn<EPS else (pit-mn)/(mx-mn)
    return np.argsort(-score), score
