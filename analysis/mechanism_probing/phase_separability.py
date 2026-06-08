import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

d = np.load('diff_cache.npz'); H1,H2,H3,H4,Y = d['H1'],d['H2'],d['H3'],d['H4'],d['Y']
def sep(eff):
    Ds=StandardScaler().fit_transform(eff)
    idx=np.random.RandomState(0).choice(len(Ds),min(4000,len(Ds)),replace=False)
    return silhouette_score(Ds[idx],Y[idx])
C_POOL="#e08a3c"; C_DENSE="#2c7fb8"
effects=[("pooled\nvs H-RDT baseline",H3-H1,C_POOL),("dense\nvs H-RDT baseline",H4-H1,C_DENSE),
         ("pooled\nvs AVP-only",H3-H2,C_POOL),("dense\nvs AVP-only",H4-H2,C_DENSE)]
labels=[e[0] for e in effects]; vals=[sep(e[1]) for e in effects]; cols=[e[2] for e in effects]
for l,v in zip(labels,vals): print(f"  {l.replace(chr(10),' '):<24} sep={v:.3f}")
fig,ax=plt.subplots(figsize=(8,4.8)); x=np.arange(4)
bars=ax.bar(x,vals,color=cols,edgecolor="black",linewidth=0.6,width=0.6)
ax.set_xticks(x); ax.set_xticklabels(labels,fontsize=9)
ax.set_ylabel("Phase separability (silhouette score)",fontsize=11)
ax.set_title("Phase separability of the LSS-induced representational change",fontsize=12)
ax.set_ylim(0,max(vals)*1.25)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,v+0.0012,f"{v:.3f}",ha="center",fontsize=9)
ax.legend(handles=[Patch(facecolor=C_POOL,edgecolor='black',label='pooled LSS'),
                   Patch(facecolor=C_DENSE,edgecolor='black',label='dense LSS')],fontsize=9,loc='upper left')
fig.text(0.5,-0.04,
 "Dense LSS yields higher phase separability than pooled LSS, consistently vs both the H-RDT baseline "
 "(EgoDex Stage-1\npretrain) and the AVP-only baseline. Absolute values are small (\u2264 0.05), bounded by "
 "per-phase reasoning-target\nsimilarity (cosine 0.70): a modest but consistent supporting result. Primary "
 "evidence is behavioral transfer.",
 ha="center",fontsize=7.5,color="#444444")
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig("/tmp/probes/phase_separability.png",dpi=200,bbox_inches="tight")
plt.savefig("/tmp/probes/phase_separability.pdf",bbox_inches="tight"); print("saved")
