import time
import pickle as pickler
import sys


#from fpylll import BKZ as BKZ_FPYLLL, GSO, IntegerMatrix
from fpylll import IntegerMatrix, GSO

from numpy import array, zeros, identity, block, matrix, linalg
from scipy.linalg import circulant
from numpy.random import shuffle
from numpy import random
import numpy as np

import time
from ntru_keygen import gen_ntru_instance_matrix, gen_ntru_instance_circulant

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


def minus_one_to_the_n(n):
     return -1 if n%2==1 else 1

def poly_prod_trunc_np(a,b,n):   #returns product of 2 polynomials represented as lists (ith iteme is the x^i's coeff)
    if type(a)==list:   #if a, b are lists, flip
        a.reverse()
        b.reverse()
    elif type(a)==np.ndarray:   #if numpy arrays, flip
        np.flip(a,0)
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

def Norm(v,m,q):   #returns norm over K=Cyclo(2n) of v \in L mod q
    n=len(v)
    G=galois_group_relative(n,m)

    p=[1]+(n-1)*[0]

    for g in G:
        v_=apply_authomorphism(v,g)
        p=np.array( poly_prod_trunc_np(p,v_,n) )%q
    return(list(p))

def to_circ(f,step=1):   #make circulant matrix out of list
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


def prepeare_for_gen_lattice(n,q,verbose=False,seed=1227):   #preprocessing before generation
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

    nH=Norm(h,n//r,q)

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
n=128
q=997
B, F, G, H = gen_lattice_norm(n,q, verbose=False, seed=1341, r=2)   #seed=1342 for dim 256 =1341 for dim 64 and 128
#B, F, G, H = gen_lattice_full(n,q,seed=1345)

print(B)
# print(F[0])
# print(G[0])
# print(H[0])
# print(np.matmul(H[0],F)%q)
