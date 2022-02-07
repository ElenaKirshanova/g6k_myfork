from __future__ import absolute_import
from __future__ import print_function
import logging
import re
import time
import pickle as pickler

from collections import OrderedDict

#from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix
from fpylll import IntegerMatrix, GSO, LLL, BKZ
from fpylll.tools.quality import basis_quality
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2

from numpy import array, zeros, identity, block, matrix, linalg
from scipy.linalg import circulant
from numpy.random import shuffle
from numpy import random
import numpy as np

import six
import numpy as np
from six.moves import range
import random
import time
import math
import sys
from ntru_keygen import gen_ntru_instance_matrix, gen_ntru_instance_circulant

def dim(a):
    if not type(a) == list:
        return []
    return [len(a)] + dim(a[0])

def egcd(a, b):
    if a == 0:
        return (b, 0, 1)
    else:
        g, y, x = egcd(b % a, a)
        return (g, x - (b // a) * y, y)

def modinv(a, m):
    g, x, y = egcd(a, m)
    if g != 1:
         raise ZeroDivisionError
    else:
         return x % m

def modinvMat(M, q):
    n, m = M.shape
    assert m==n

    invs = q * [None]
    for i in range(1, q):
        try:
            invs[i] = modinv(i, q)
            assert((i*invs[i]) % q == 1)
        except ZeroDivisionError:
            pass

    R = np.block([[M, np.identity(n, dtype="long")]])
    #print(R)

    # i-th column
    for i in range(n):
        #print(i, q, R)
        # Find a row with an invertible i-th coordinate
        for j in range(i, n+1):
            if j == n:
                raise ZeroDivisionError

            # Normalize the row and swap it with row j
            if invs[R[j,i]] is not None:
                R[j] = (R[j] * invs[R[j,i]]) % q

                if j > i:
                    R[i], R[j] = R[j], R[i]
                break

        # Kill all coordinates of that column except at row j
        for j in range(n):
            if i==j: continue
            R[j] = (R[j] -  R[i] * R[j, i]) % q

    #print(i, R)

    Minv = R[:,n:]
    return Minv

# def modMatInv(A,p):       # Finds the inverse of matrix A mod p
#   n=len(A)
#   A=matrix(A)
#   adj=np.zeros(shape=(n,n), dtype=int)
#   for i in range(0,n):
#     for j in range(0,n):
#       adj[i][j]=((-1)**(i+j)*int(round(linalg.det(minor(A,j,i)))))%p
#   return (modInv(int(round(linalg.det(A))),p)*adj)%p
#
# def modInv(a,p):          # Finds the inverse of a mod p, if it exists
#   for i in range(1,p):
#     if (i*a)%p==1:
#       return i
#   raise ValueError(str(a)+" has no inverse mod "+str(p))
#
# def minor(A,i,j):    # Return matrix A with the ith row and jth column deleted
#   A=np.array(A)
#   minor=np.zeros(shape=(len(A)-1,len(A)-1), dtype=int)
#   p=0
#   for s in range(0,len(minor)):
#     if p==i:
#       p=p+1
#     q=0
#     for t in range(0,len(minor)):
#       if q==j:
#         q=q+1
#       minor[s][t]=A[p][q]
#       q=q+1
#     p=p+1
#   return minor

def minus_one_to_the_n(n):
     return -1 if n%2==1 else 1

# def fft(n,w,p):
#
#     if n==1: return p
#
#     a=fft(n//2,w*w,p[0::2])#even powers
#     b=fft(n//2,w*w,p[1::2])#odd powers
#
#     ret=[None]*n
#     x=1
#     for k in range(n/2):
#         ret[k]=a[k]+x*b[k]#p(a)
#         ret[k+n/2]=a[k]-x*b[k]#p(−a)
#         x*=w
#
#     return ret

# def poly_prod_fft(n,p1,p2):
#     #nth principal root of unity
#     w=complex(cos(2*pi/n),sin(2*pi/n))
#
#     #forward transform to point value
#     a=fft(n,w,p1)
#     b=fft(n,w,p2)
#
#     #convolution
#     p3=[a[i]*b[i]for i in range(n)]
#
#     #reversetransformtogetbackcoefficients
#     final=fft(n,w**(-1),p3)
#     return[int((x/n).real) for x in final]
#
# def poly_prod_trunc_fft(s1,s2,n):   #doesn't work well due to rounding
#     p=poly_prod_fft(2*len(s1),s1,s2)
#     tmp=n*[0]
#     for i in range(len(tmp)):   #truncated polynomial mod x^n+1
#         tmp[i]=p[i]-p[n+i]
#     return np.array(tmp)

# def poly_prod_trunc(s1,s2,n):   #multiplies s1 and s2 modulo x^n+1
#     #s1, s2 = list(s1), list(s2)
#
#     res_ = [0]*(n)
#     for o1,i1 in enumerate(s1):
#         for o2,i2 in enumerate(s2):
#             #print(i1,i2)
#             ind=(o1+o2)
#             res_[(ind)%n] += i1*i2*minus_one_to_the_n((ind)//(n))
#     res=np.ndarray(shape=n, dtype=int)
#     for i in range(n):
#         res[i]=res_[i]
#     return(res)

def poly_prod_trunc_np(a,b,n):
    if type(a)==list:   #if a, b not poly1d
        a.reverse()
        b.reverse()
    elif type(a)==np.ndarray:
        np.flip(a,0) #- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        np.flip(b,0)

    a, b = np.poly1d(a), np.poly1d(b)
    c=a*b
    c=[i for i in reversed(c.c)]

    while(len(c)<2*n):
        c.append(0)

    c_=[ c[i]-c[n+i] for i in range(n)]
    return c_

def sign1(x):   #singum, but sign(0)=1
    return 1 if x==0 else x//abs(x)

# def galois_group(n):   #returns Galois group of n-th cyclotomic field, n is power of 2
#     v=[i for i in range(n)]
#     W=[]
#     for morph in range(1,2*n,2):
#         w=n*[0]
#         for i in range(n):
#             i_=(i*morph) % (2*n)
#             sign= 1 if i_<n else -1
#             i_=i_%n
#             w[i_]=sign*v[i]
#         W.append(w)
#     return W

def galois_group_relative(n,m):   #returns Galois group of L/K where L=Cyclo(2n) K=Cyclo(2m)
    r=round( n//m )
    v=[i for i in range(n)]
    W=[]
    for morph in range(1,2*n,2*m):
        w=n*[0]
        for i in range(n):
            i_=(i*morph) % (2*n)
            sign= 1 if i_<n else -1
            i_=i_%n
            w[i_]=sign*v[i]
        W.append(w)
    return W

def apply_authomorphism(v, sigma):   #applies sigma \in Gal to v
    w=[ sign1(sigma[i])*v[abs( sigma[i] )] for i in range(n)]
    return w

def poly_conj(v):
    for i in range(1,len(v),2):
        v[i]=-v[i]
    return(v)

def Norm(v,m):   #returns norm over K=Cyclo(2n) of v \in L
    n=len(v)
    G=galois_group_relative(n,m)

    p=[1]+(n-1)*[0]

    for g in G:
        v_=apply_authomorphism(v,g)
        p=np.array( poly_prod_trunc_np(p,v_,n) )
    return(list(p))

def balance_matrix(M,q):
    M=M%q
    sh=M.shape
    lensh=len(sh)
    is_vect=False
    if lensh==1:
        is_vect=True
        r, c = 1, sh[0]
        raise Exception("Vectors not supported.")
    elif lensh==2:
        r, c = sh[0], sh[1]
    else:
        raise Exception("Critical error in balance matrix!")

    N=np.ndarray(shape=(r,c), dtype=int)
    for row in range(r):
        for col in range(c):
            if not is_vect:
                tmp=int(M[row,col])%q
            else:
                tmp=int(M[col])%q
            N[row,col]=tmp if tmp<q//2 else -q+tmp
    return N



def to_circ(f,step=1):
    n=f.shape[0]
    B=[]
    for row in range(n):
        tmp=[0]*n
        for i in range(row,n):
            tmp[i]=f[i-row]
        for j in range(row):
            tmp[j]=-f[n-row+j]
        B.append(tmp)
    return np.array(B)

# def cyclo_trace_n2(f, div=False):
#     #trace to Cyclo_(n//2) field
#     n=f.shape[0]
#     tmp=np.ndarray(shape=(n),dtype=int)
#     tmp.fill(0)
#
#     if div:
#         tmp_=[((i+1)%2)*f[i] for i in range(n)]
#         for i in range(len(tmp_)):
#             tmp[i]=tmp_[i]
#     else:
#         tmp_=[2*((i+1)%2)*f[i] for i in range(n)]
#         for i in range(len(tmp_)):
#             tmp[i]=tmp_[i]
#     return tmp

# def cyclo_norm_n2(f):
#     #norm to Cyclo_(n//2) field
#     nn=f.shape[0]
#     tmp=[0]*n
#     tmp=np.ndarray(shape=(n),dtype=int)
#     tmp.fill(0)
#
#     for i in range(n):
#         tmp_=[(-1)**i*f[i] for i in range(n)]
#         for i in range(len(tmp_)):
#             tmp[i]=tmp_[i]
#
#     return poly_prod_trunc(tmp,f,n)


def prepeare_for_gen_lattice(n,q,verbose=False,seed=1227):
    d=2*n
    print('n, d are:', n, d)
    sigmasq=0.666

    _, F, G = gen_ntru_instance_circulant(n, q, sigmasq, seed)

    F=to_circ(F[0])
    G=to_circ(G[0])

    print('Inverting Matrix...')
    F_inv=modinvMat(F,q)
    print('Matrix Inverted')
    H=np.matmul(G,F_inv)


    H=to_circ(H[0])

    #print('lol', balance_matrix(np.matmul(H,F),q) )
    # H=np.array(H)
    # G=np.array(G)
    # F=np.array(G)


    if verbose:
        print(F)
        print('- - -')
        print(G)
        print('- - -')
        print(H)
        print('- - -')


    return F, G, H

def gen_lattice_full(n,q,verbose=False,seed=1227):
    F, G, H = prepeare_for_gen_lattice(n,q,verbose, seed)

    B=np.ndarray(shape=[2*n,2*n], dtype=int)
    B.fill(0)

    for i in range(2*n):
        B[i,i]=q if i<n else 1

    for row in range(n,2*n):
        for col in range(n):
            B[row,col]=H[row-n,col]
    return B, F, G, H


def gen_lattice_norm(n,q,verbose=False,seed=1227,r=2):  #prepeare norm matrix where descent index is r
    F, G, H = prepeare_for_gen_lattice(n,q,verbose, seed)
    tmp=[H[0][i] for i in range(n)]
    h=np.ndarray(shape=n, dtype=int)
    h.fill(0)
    for i in range(n):
        h[i]=tmp[i]

    nH=Norm(h,n//r)

    del_indexes=[]
    for i in range(n):
        if i%r!=0:
            del_indexes.append(i)

    nH=np.delete(nH,del_indexes, axis=0)

    nH=to_circ(nH)
    #nH=balance_matrix(nH,q)

    if verbose:
        print(nH)
        print('- - -')

    B=np.ndarray(shape=(2*n//r,2*n//r), dtype=int)
    B.fill(0)
    for i in range(2*n//r):
        B[i,i]=q if i<n//r else 1

    for row in range(n//r,2*n//r):
        for col in range(n//r):
            B[row,col]=nH[row-n//r,col]

    return B, F, G, H

np.set_printoptions(threshold=sys.maxsize,linewidth=202)
n=16
q=997
B, F, G, H = gen_lattice_norm(n,q, verbose=False, seed=1341, r=2)   #seed=1342 for dim 256 =1341 for dim 64 and 128
#B, F, G, H = gen_lattice_full(n,q,seed=1345)

print(B)
print(F[0])
print(G[0])
print(H[0])
print(np.matmul(H[0],F)%q)

np.set_printoptions(threshold=sys.maxsize,linewidth=202)
n=128
q=997
B, F, G, H = gen_lattice_norm(n,q, verbose=False, seed=1341, r=2)   #seed=1342 for dim 256 =1341 for dim 64 and 128

print(B)
print(F[0])
print(G[0])
print(H[0])
print(np.matmul(H[0],F)%q)
