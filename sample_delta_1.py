# %%
import jax.numpy as jnp
import numpy as np
from math import factorial as fac
from math import log
from pprint import pprint
import jax
from functools import partial
import itertools
from tqdm import tqdm

# %%
def sample_cbd(N, eta):
    '''
    Sampling from the centered binomial distribution.
    N: number of Samples
    eta: samples lie in the interval [-eta, eta]
    '''
    c = np.random.binomial(2*eta,0.5,N)-eta
    return c

def sample_uniform(q, size):
    return np.random.randint(0, q, size)

# %%
# jnp.convolve always outputs floats (error in code), so we rewrite it for integers

@partial(jax.jit, static_argnames=['preferred_element_type'])
def int_convolve(x, y, preferred_element_type=jnp.int32):
  if jnp.ndim(x) != 1 or jnp.ndim(y) != 1:
    raise ValueError(f"{op}() only support 1-dimensional inputs.")

  assert x.dtype == y.dtype
  if len(x) == 0 or len(y) == 0:
    raise ValueError(f"{op}: inputs cannot be empty, got shapes {x.shape} and {y.shape}.")

  out_order = slice(None)
  if len(x) < len(y):
    x, y = y, x
  y = jnp.flip(y)
  padding = [(y.shape[0] - 1, y.shape[0] - 1)]

  result = jax.lax.conv_general_dilated(x[None, None, :], y[None, None, :], (1,), 
                                        padding, precision=None, preferred_element_type=preferred_element_type)
  return result[0, 0, out_order] 

# %%
def poly_mult(p1,p2,q):
    '''
    Multiplication of two polynomials p1 and p2 in R_q
    p1 :: np.array of coefficients length n
    p2 :: np.array of coefficients length n
    q :: modulus of R_q
    '''
    n = len (p1)
    conv = jnp.concatenate((int_convolve(p1,p2),jnp.array([0])))
    conv_q = (conv[:n] - conv[n:]) % q
    return conv_q

def poly_mult_vec(p,w,q):
    '''
    Multiplication of a polynomial p with vector of sample polynomials w
    p :: (n) dimensional polynomial
    w :: (N,n) dimensional vector
    '''
    return jax.vmap(lambda w: poly_mult(p,w,q))(w)

def sprod_poly(v,w,q):
    '''
    Scalar product of two vectors v and w in R_q^k
    '''
    assert v.ndim == 2 and w.ndim == 2
    vec = jax.vmap(lambda x,y: poly_mult(x, y, q))(v,w)
    return vec.sum(axis=0)

def sprod_vec(v,w,q):
    '''
    scalar product of sample vectors of polynomials v and w in R_q^k^N
    '''
    vec = jax.vmap(lambda v,w: sprod_poly(v,w,q))(v,w)
    return vec

def mat_vec_mult(A,s,q):
    '''
    Multiplication of a matrix A in R_q^(k x k) with a vector s in R_q^k
    '''
    m = jax.vmap(lambda x: sprod_poly(x, s, q))(A)
    return m

def vec_add(a,b,q):
    '''
    Addition of two vectors of polynomials a and b in R_q^k
    a :: np.array of vector of coefficients
    b :: np.array of vector of coefficients
    q :: modulus of R_q
    '''
    return ((a+b)%q)

def vec_minus(a,b,q):
    '''
    Difference of two vectors of polynomials a and b in R_q^k
    a :: np.array of vector of coefficients
    b :: np.array of vector of coefficients
    q :: modulus of R_q
    '''
    return ((a-b)%q)

# %%
def compress(x,q,d):
    '''
    compression of integer x from modulus q to modulus 2^d
    '''
    return round (x * (2**d) / q) % (2**d)

def decompress(x,q,d):
    '''
    decompression of integer x from modulus q to modulus 2^d
    '''
    return round (x * q / (2**d)) % q


# %%
def compress_poly(p,q,d):
    '''
    Compression on polynomials in R_q
    '''
    return jax.vmap(lambda p: compress(p,q,d))(p)

def decompress_poly(p,q,d):
    '''
    Decompression on polynomials in R_q
    '''
    return jax.vmap(lambda p: decompress(p,q,d))(p)

# %%
def centered_mod(p,q):
    p = p % q
    c = jnp.where(p <= (q / 2), p, p-q)
    return c

def centered_mod_norm(p,q):
    return jnp.abs(centered_mod(p,q))

def poly_norm(p,q):
    return jnp.max(centered_mod_norm(p,q))

# %%
class Kyber:
    def __init__(self, n, k, q, du, dv, eta1, eta2=None):
        '''
        eta1 :: variable of beta_eta distribution (centered binomial distribution), values lie in [-eta1,eta1]
        eta2 :: variable of beta_eta distribution (centered binomial distribution), values lie in [-eta2,eta2]
        q :: modulus of R_q
        n :: (x^n+1) is polynomial factored out in R_q, ie n is the number of coefficients of polynomials in R_q
        k :: vector/matrix dimension, ie R_q^k
        '''
        if eta2 is None:
            eta2 = eta1
        self.n = n
        self.k = k
        self.q = q
        self.du = du  # 2^(bits in the first ciphertext)
        self.dv = dv  # 2^(bits in the second ciphertext)
        self.eta1 = eta1
        self.eta2 = eta2

    def sample_key_gen(self, N):
        '''
        sampling of key generation algortihm of Kyber
        N :: number of samples taken
        '''
        (n,k,q,eta1) = (self.n,self.k,self.q,self.eta1)
        s = sample_cbd((N,k,n),eta1)
        e = sample_cbd((N,k,n),eta1)
        A = sample_uniform(q, (N,k,k,n))
        t = jax.jit(jax.vmap(lambda A, s, e: vec_add (mat_vec_mult(A,s,q), e, q))) (A,s,e)
        return (A, t, s, e)

    def msg_times_q_half(self, m): 
        to_mod_q = jnp.zeros((self.n), dtype=jnp.int32)
        to_mod_q = to_mod_q.at[0].set((self.q+1) // 2)
        qm = poly_mult(to_mod_q, m, self.q)
        return qm  # = round(q/2) *m
    
    def sample_encryption(self, N, A, t, m):
        '''
        sampling of encryption algortihm of Kyber with compression!
        N :: number of samples taken
        A and t :: public key
        m :: message to be encoded
        '''
        (n,k,q,du,dv,eta1,eta2) = (self.n,self.k,self.q,self.du,self.dv,self.eta1,self.eta2)
        assert m.shape == (n,)
        
        r = sample_cbd((N,k,n),eta1)
        e1 = sample_cbd((N,k,n),eta1) # in R_q^k
        e2 = sample_cbd((N,n),eta1) # in R_q
        At = A.transpose((0,2,1,3))
        qm = self.msg_times_q_half(m)
        u = jax.vmap(lambda A: jax.vmap(lambda r, e1: vec_add (mat_vec_mult(A,r,q), e1, q)) (r,e1))(At)
        v = jax.vmap(lambda t: jax.vmap(lambda r, e2: vec_add(vec_add(sprod_poly(t, r, q), e2, q), qm, q))(r, e2))(t)
        comp_u = compress_poly(u, q, du) 
        comp_v = compress_poly(v, q, dv)
        return (comp_u, comp_v)

    def decrypt(self,N,A,t,s,e,comp_u,comp_v):
        dc_v = decompress_poly(comp_v, self.q, self.dv)
        dc_u = decompress_poly(comp_u, self.q, self.du)
        st_dcu = jax.vmap(lambda s, dc_u: jax.vmap(lambda dc_u: sprod_poly(s, dc_u, self.q))(dc_u))(s, dc_u)
        return vec_minus(dc_v, st_dcu,self.q)

    def inner_P(self, N, A, t, s, e, m): # for a given A,t,s: norm_w is a vector of booleans of shape (N1,N2)
        (eta1, eta2, q, n, k, du, dv) = (self.eta1, self.eta2, self.q, self.n, self.k, self.du, self.dv)
        q4 = (q+2) // 4
        qm = self.msg_times_q_half(m)   # = round(q/2) *m
        (comp_u,comp_v) = self.sample_encryption(N, A, t, m)
        dec = self.decrypt(N,A,t,s,e,comp_u,comp_v)
        w = vec_minus(dec, qm, q) # == 0 if decrypt(encrypt(m)) = m, != 0 otherwise
        norm_w = jax.vmap(lambda x: jax.vmap(lambda y: poly_norm(y,q) >= q4)(x))(w)
        return norm_w

    def inner_P_original(self, N, A, t, s, e, m): # for a given A,t,s: norm_w is a vector of booleans of shape (N1,N2)
        (eta1, eta2, q, n, k, du, dv) = (self.eta1, self.eta2, self.q, self.n, self.k, self.du, self.dv)
        (comp_u,comp_v) = self.sample_encryption(N, A, t, m)
        dec = self.decrypt(N,A,t,s,e,comp_u,comp_v)
        compress_dec = compress_poly(dec, q, 1)
        w = vec_minus(compress_dec, m, q) # == 0 if decrypt(encrypt(m)) = m, != 0 otherwise
        res = jax.vmap(lambda x: jax.vmap(lambda y: poly_norm(y,q) != 0)(x))(w)
        return res


# %%
def msg_space(n):
    M = np.array(list(itertools.product([0, 1], repeat=n)))
    return M


# %%
def calc_estimation(kyber, N1, N2):
    (A,t,s,e) = kyber.sample_key_gen(N1)
    # m = jnp.zeros((kyber.n), dtype=jnp.int32)
    calc_fun = jax.jit(kyber.inner_P, static_argnums=[0])
    inner_truth = jax.vmap(lambda m: calc_fun(N2, A, t, s, e, m))(msg_space(kyber.n))
    inner_mean = inner_truth.mean(axis=2) # this calculates the inner probabilities for every message and all key_gen_samples
    max_of_inner = inner_mean.max(axis = 0)
    estimation = max_of_inner.mean()
    return estimation

def calc_original(kyber, N1, N2):
    (A,t,s,e) = kyber.sample_key_gen(N1)
    calc_fun = jax.jit(kyber.inner_P_original, static_argnums=[0])
    inner_truth = jax.vmap(lambda m: calc_fun(N2, A, t, s, e, m))(msg_space(kyber.n))
    inner_mean = inner_truth.mean(axis=2) # this calculates the inner probabilities for every message and all key_gen_samples
    max_of_inner = inner_mean.max(axis = 0)
    estimation = max_of_inner.mean()
    return estimation

# %%
'''
estimation of delta with sampling:
sample key generation => s, A, t
sample encryption => r, e1, e2
calculate compression/decompression
maximum over smaller key spaces, for example 2^16
samples for probability in maximum (calculation)
'''
def estimate(kyber):
    estimates = np.empty([10], dtype=np.float64)
    for i in tqdm(range(10)):
        estimates[i] = calc_estimation(kyber, 100, 1000)
    mean = estimates.mean()
    return mean


'''
estimation of original kyber decryption error with sampling:
'''
def estimate_original(kyber):
    estimates = np.empty([10], dtype=np.float64)
    for i in tqdm(range(10)):
        estimates[i] = calc_original(kyber, 100, 1000)
    mean = estimates.mean()
    return mean

# %%
#test = Kyber(n=8,k=2,q=103,eta1=2,du=5,dv=2)
#N1 = 100
#N2 = 1000

#estimate(test)




# %%


# %%



