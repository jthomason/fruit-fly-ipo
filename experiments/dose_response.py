import sys; sys.path.insert(0, ".")
import json,numpy as np
from flybrain import FlyBrain
S=json.load(open('sets630.json')); B=FlyBrain(version=630)
mn9=B.ix(S['ids_mn9']); dn=B.ix([720575940622537477,720575940627827697]); sug=B.ix(S['neu_sugar']); bit=B.ix(S['neu_bitter'])
for r in [10,25,50,100,150,200,300]:
    a=[];b=[]
    for tr in range(5):
        B.reset(); B.run(30,[]); c=B.run(100,[(sug,r)]); a.append(c[mn9].sum())
        B.reset(); B.run(30,[]); c=B.run(100,[(bit,r)]); b.append(c[dn].sum())
    print(r,'Hz  sugar->MN9 spikes/100ms',np.mean(a),'  bitter->DNge067',np.mean(b))
