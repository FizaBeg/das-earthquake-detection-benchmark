import gzip, struct, sys, numpy as np
from tfrecord import example_pb2
def iterate(path):
    with gzip.open(path,'rb') as f:
        while True:
            h=f.read(8)
            if len(h)<8: break
            n=struct.unpack('<Q',h)[0]; f.read(4)
            d=f.read(n); f.read(4)
            ex=example_pb2.Example(); ex.ParseFromString(d)
            fe=ex.features.feature
            x=np.frombuffer(fe['inputs'].bytes_list.value[0],dtype=np.float32)
            y=np.frombuffer(fe['labels'].bytes_list.value[0],dtype=np.float32)
            yield x,y
if __name__=='__main__':
    ys=[];sh=set();st=[]
    for i,(x,y) in enumerate(iterate(sys.argv[1])):
        sh.add(x.size); ys.append(y[0])
        if i<5: st.append((x.min(),x.max(),x.std()))
    ys=np.array(ys); print('n',len(ys),'sizes',sh,'pos',ys.sum(),'neg',(ys==0).sum()); print(st)
