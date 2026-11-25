import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
X=np.load('/home/claude/data/npy/test_X.npy'); y=np.load('/home/claude/data/npy/test_y.npy')
Xp=np.load('/home/claude/data/npy/test_Xp.npy')
# pick examples deterministically: first two events and first two noise
ie=np.where(y==1)[0][:2]; inn=np.where(y==0)[0][:2]
fig,ax=plt.subplots(2,3,figsize=(10,6.2),constrained_layout=True)
t=np.arange(695)/25.0
for r,(i,lab) in enumerate([(ie[0],'Earthquake'),(inn[0],'Noise')]):
    a=ax[r,0]; im=a.imshow(X[i].astype(float),aspect='auto',cmap='seismic',vmin=-1.9,vmax=1.9,extent=[0,t[-1],288,0])
    a.set_title(f'({"ad"[r]}) {lab}: full resolution (288 ch)',fontsize=9); a.set_ylabel('Channel index'); a.set_xlabel('Time (s)')
    a=ax[r,1]; a.imshow(Xp[i],aspect='auto',cmap='seismic',vmin=-1.2,vmax=1.2,extent=[0,347*2/25,48,0])
    a.set_title(f'({"be"[r]}) {lab}: model input (48 ch)',fontsize=9); a.set_xlabel('Time (s)'); a.set_ylabel('Pooled channel')
    a=ax[r,2]
    for k,c in enumerate([40,140,240]):
        a.plot(t,0.5*X[i,c].astype(float)+2.5*k,lw=0.4,color="k")
    a.set_yticks([0,2.5,5]); a.set_yticklabels(['ch 40','ch 140','ch 240'])
    a.set_title(f'({"cf"[r]}) {lab}: single-channel traces',fontsize=9); a.set_xlabel('Time (s)')
fig.colorbar(im,ax=ax[:,0],shrink=0.6,label='Normalized strain rate')
plt.savefig('/home/claude/paper/figures/fig1_data.pdf'); plt.savefig('/home/claude/paper/figures/fig1_data.png',dpi=130)
print(ie,inn)
