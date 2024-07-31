import numpy as np

def prior(x, kappa, Nt=None, addCluster=None):
    if Nt is None:
        Nt = x['z'].shape[0]
    p = x['mu'].shape[1]
    K = x['K']
    nu0 = Ng = x['w'] * Nt
    if np.all((nu0 * kappa - p - 1) > 0):
        Lambda0 = x['sigma'].copy()
        for i in range(K):
            Lambda0[i] *= (kappa * nu0[i] - p - 1)
    else:
        raise ValueError("Can't proceed. Prior nu0 is negative for cluster(s). Try increasing kappa.")
    
    Omega0 = np.zeros((K, p, p))
    for i in range(K):
        Omega0[i] = np.eye(p)
        if p == 1:
            dS = x['sigma'][i]
            dO = Omega0[i]
        else:
            dS = np.linalg.det(x['sigma'][i])
            dO = np.linalg.det(Omega0[i])
        k = (dO / dS) ** (1 / p)
        Omega0[i] *= k
        Omega0[i] = np.linalg.inv(Omega0[i] * Ng[i] * kappa)
    
    nu0 *= kappa
    Mu0 = x['mu']
    lambda_ = x['lambda']
    w0 = x['w'] * Nt
    
    if addCluster is not None:
        for i in range(K, K + addCluster):
            S = np.cov(Mu0, rowvar=False)
            Lam = np.zeros((K + 1, p, p))
            om = np.zeros((K + 1, p, p))
            Mu0 = np.vstack([Mu0, Mu0.mean(axis=0)])
            for j in range(K):
                om[j] = Omega0[j]
                Lam[j] = Lambda0[j]
            om[K] = np.eye(p)
            Lam[K] = np.eye(p)
            np.fill_diagonal(Lam[K], np.diag(S))
            np.fill_diagonal(om[K], np.diag(S))
            if p == 1:
                dS = Lam[K]
                dO = om[K]
            else:
                dS = np.linalg.det(Lam[K])
                dO = np.linalg.det(om[K])
            k = (dO / dS) ** (1 / p)
            om[K] *= k
            Omega0 = om
            Lambda0 = Lam
            nu0 = np.append(nu0, p + 2)
            w0 = np.append(w0, 1)
            K += 1
    
    prior = {
        'Mu0': Mu0,
        'Lambda0': Lambda0,
        'Omega0': Omega0,
        'w0': w0,
        'nu0': nu0,
        'nu': x['nu'],
        'lambda': lambda_,
        'K': K
    }
    prior['lambda'] = lambda_
    
    return prior

# Example usage (ensure you have the data structure as per your requirement)
# x = {
#     'z': np.array(...),
#     'mu': np.array(...),
#     'sigma': np.array(...),
#     'w': np.array(...),
#     'K': ...,
#     'nu': ...,
#     'lambda': ...
# }
# kappa = ...
# result = prior(x, kappa)
# print(result)
