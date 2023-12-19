from math import log
from Kyber_failure import p2_cyclotomic_error_probability
from proba_util import build_mod_switching_error_law

class KyberParameterSet:
    def __init__(self, n, m, ks, ke,  q, rqk, rqc, rq2, ke_ct=None):
        if ke_ct is None:
            ke_ct = ke
        self.n = n
        self.m = m
        self.ks = ks     # binary distribution for the secret key
        self.ke = ke    # binary distribution for the ciphertext errors
        self.ke_ct = ke_ct    # binary distribution for the ciphertext errors
        self.q = q
        self.rqk = rqk  # 2^(bits in the public key)
        self.rqc = rqc  # 2^(bits in the first ciphertext)
        self.rq2 = rq2  # 2^(bits in the second ciphertext)


def summarize(ps):
    print ("params: ", ps.__dict__)
    #print ("com costs: ", communication_costs(ps))
    F, f = p2_cyclotomic_error_probability(ps)
    print ("failure: %.1f = 2^%.1f"%(f, log(f + 2.**(-300))/log(2)))


if __name__ == "__main__":
    # Parameter sets
    ps_light = KyberParameterSet(256, 2, 3, 3, 3329, 2**12, 2**10, 2**4, ke_ct=2)
    ps_recommended = KyberParameterSet(256, 3, 2, 2, 3329, 2**12, 2**10, 2**4)
    ps_paranoid = KyberParameterSet(256, 4, 2, 2, 3329, 2**12, 2**11, 2**5)

    # Analyses
    print ("Kyber512 (light):")
    print ("--------------------")
    summarize(ps_light)
    print ()

    print ("Kyber768 (recommended):")
    print ("--------------------")
    summarize(ps_recommended)
    print ()

    print ("Kyber1024 (paranoid):")
    print ("--------------------")
    summarize(ps_paranoid)
    print ()
