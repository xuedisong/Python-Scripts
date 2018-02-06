# -*- coding: utf-8 -*-
#杨辉三角
def fname(n):
    L1=[1]
    for k in range(1,n+1):
        if k==1:
            L1[0]=1
            yield(L1)
        else:
            if k>2:
                for i in range(1,k-1):
                    L1[i]=L1[i-1]+L1[i]
            L1[k-1]=1
            yield(L1)
for i in fname(10):
    print i
