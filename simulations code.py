import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy.integrate import quad
from functools import reduce
from itertools import product
import matplotlib.pyplot as plt

norm = np.linalg.norm
pi = np.pi
cos = np.cos
sin = np.sin

def psi(i, h):
    def _psi(x):
        val = abs((x-i*h)/h)     
        if val <= 1:
            return 1 - val
        else:
            return 0
    return _psi


def get_M(I):
    h = 1 / (I + 1)
    subd = [h / 6 for _ in range(I - 1)]
    main = [2 * h / 3 for _ in range(I)]
    return diags([subd, main, subd], offsets=[-1, 0, 1])

def get_D(I):
    h = 1 / (I + 1)
    subd = [-1/h for _ in range(I - 1)]
    main = [2/h for _ in range(I)]
    return diags([subd, main, subd], offsets=[-1, 0, 1])

def get_F(f, funcs, divs):
    I = len(divs) - 2
    res = []
    for fx, fy in f:
        tmpx = [quad(lambda t: fx(t) * funcs[i](t), divs[i-1], divs[i+1])[0] for i in range(1, I + 1)]
        tmpy = [quad(lambda t: fy(t) * funcs[i](t), divs[i-1], divs[i+1])[0] for i in range(1, I + 1)]
        res.append((np.array(tmpx), np.array(tmpy)))
    return res


def get_funcs(I):
    h = 1 / (I + 1)
    funcs = dict()
    for i in range(1, I + 1):
        funcs[i] = psi(i, h)
    return funcs

def FN(F, D, M, SR):
    I = D.shape[0]
    S, R = SR
    def _F(V):
        sum1 = reduce(lambda old, nw: old + (V.T@nw[1])*nw[0], F, np.zeros(I))
        sum2 = reduce(lambda old, nw: (V.T@D@nw[1])*(M@nw[0]) + (V.T@M@nw[1])*(D@nw[0]), zip(S, R), np.zeros(I))
        return sum1 - sum2
    return _F

def GN(F, D, M, SR):
    I = D.shape[0]
    S, R = SR
    def _G(V):
        sum1 = reduce(lambda old, nw: old + (V.T@nw[0])*nw[1], F, np.zeros(I))
        sum2 = reduce(lambda old, nw: (V.T@D@nw[0])*(M@nw[1]) + (V.T@M@nw[0])*(D@nw[1]), zip(S, R), np.zeros(I))
        return sum1 - sum2
    return _G

def naif_A(M, D):
    n = D.shape[0]
    M, D = M.toarray(), D.toarray()
    ret = np.zeros((n, n, n, n))
    for i, j, k, l in product(range(n), repeat=4):
        ret[i][j][k][l] = D[i][k]*M[j][l] + M[i][k]*D[i][l]
    return csc_matrix([[ret[i][j][k][l] for i, j in product(range(n), repeat=2)] for k, l in product(range(n), repeat=2)])

def naif_F(F, n):
    ret = np.zeros(shape=(n, n))
    for ix, iy in F:
        for i in range(n):
            for j in range(n):
                ret[i][j] += ix[i]*iy[j]
    return csc_matrix([ret[i][j] for i, j in product(range(n), repeat=2)]).T

f1 = [(lambda x: cos(2*pi*x), lambda y: cos(4*pi*y))]

f2 = [(lambda x: sin(pi*x)*sin(pi*x), lambda y: sin(2*pi*y)),
      (lambda x: sin(10*pi*x), lambda y: sin(pi*y))]

class Naif:
    
    def __init__(self, f=[], I=10):
        self.h = 1 / (I + 1)
        self.I = I
        divs = np.array([i * self.h for i in range(0, I + 2)])
        self.F = naif_F(get_F(f, get_funcs(I), divs), I)
        self.A = naif_A(get_M(I), get_D(I))
        self.res = self.solve()
        
    def solve(self):
        return spsolve(self.A, self.F)
        

class Glouton:
    
    def __init__(self, f=[], I=10, n=10, m=50):
        self.h = 1 / (n + 1)
        self.I = I
        self.n = n
        self.m = m
        divs = np.array([i * self.h for i in range(0, I + 2)])
        funcs = get_funcs(I)
        M = get_M(I)
        D = get_D(I)
        self.M = M
        self.D = D
        self.Mf = lambda v: (v.T@D@v)*M+(v.T@M@v)*D
        self.F = get_F(f, funcs, divs)
        self.SR = [], []
        self.err = [0 for _ in range(m + 1)]
        self.res = self.solve()
        
    def solve(self):
        I = self.I
        n = self.n
        for _ in range(n):
            self.next()
        S, R = self.SR
        ret = []
        for ind in range(1, n + 1):
            ret.append([sum([S[k][j]*R[k][i] for k in range(ind)]) for i, j in product(range(I), repeat=2)])
        self.mem = ret
        return ret[-1]
        
    def next(self):
        ## M(s_n^m) r_n^{m+1} = F(s_n^m)
        ## M(r_n^{m+1}) s_n^{m+1} = F(r_n^{m+1})
        S, R = self.SR
        _s = np.random.rand(self.I)
        _r = np.random.rand(self.I)
        for ind in range(1, self.m + 1):
            s_tmp, r_tmp = _s, _r
            _s, _r = self.next_fixed(_s, _r)
            curr_err = max(norm(s_tmp-_s), norm(r_tmp-_r))
            self.err[ind] = max(self.err[ind], curr_err)
        S.append(_s)
        R.append(_r)
        self.SR = S, R
            
    def next_fixed(self, _s, _r):
        Fn = FN(self.F, self.D, self.M, self.SR)
        Gn = GN(self.F, self.D, self.M, self.SR)
        null = np.zeros(self.I)
        
        if np.array_equal(_s, null):
            assert np.array_equal(Fn(_s), null)
            _r = np.zeros(self.I)
        else:
            Ms = self.Mf(_s)
            _r = spsolve(Ms, Fn(_s))
            
        if np.array_equal(_r, null):
            assert np.array_equal(Gn(_r), null)
            _s = np.zeros(self.I)
        else:
            Mr = self.Mf(_r)
            _s = spsolve(Mr, Gn(_r))

        return _s, _r
    
#############################################################################################################
#############################################################################################################

def resolution_glouton(I=10, n=10, m=10, f=[]):
    """parametres d'entrees:
            I: represente le meme I du probleme
            n: nombre d'iterations de l'algorithme Glouton
            m: nombre d'iterations de la methode du point fixe
            f: liste de tuples qui décrit le terme 'f'
                    EXEMPLE: si on veut decrire f(x, y) = a1(x)b1(y)+a2(x)b2(y)
                    on prend f = [(x-->a1(x), y-->b1(x)), (x-->a2(x), y-->b2(x))]
                    
    La fonction renvoie un tableau (dimension 1) des valeurs de la solution
    sur le maillage, un choix a ete fait sur la dimension pour comparer plus
    facilement les methodes. Si la solution est (x, y) --> u(x, y)
    On renvoit (les x_i sont les points du maillage)
                [u(x_1, x_1),...,u(x_1, x_I)
                ,u(x_2, x_1),...,u(x_2, x_I),
                ,...
                u(x_I, x_1),...,u(x_I, x_I)]
    """
    return Glouton(f, I, n, m).res

def resolution_naif(func=[]):
    """parametres d'entrees
            I: represente le meme I du probleme
            f: representation de la fonction f
            (voir Docstrings de resolution_glouton() )
    La fonction renvoie la solution approchee 'u_h'
    sous la meme forme que resolution_glouton()
    """
    return Naif(func, I=10).res

def test_convergence_m(func, start, end):
    """parametres d'entree:
            func: liste de tuples pour decrire la fonction 
            (voir docstring de resolution_glouton)
            start: debut du domaine considere (pour les valeurs de m)
            end: fin
    on represente la courbe (valeurs de m) contre 
    (erreurs maximales, methode du point fixe)
    """
    xaxis = range(start, end + 1)
    errs = Glouton(f=func, m=end).err[start:]
    plt.plot(xaxis, errs)
    plt.xlabel("m")
    plt.ylabel("Erreur maximale")
    plt.title(f"Convergence en m, P={len(func)}")
    plt.show()

def test_convergence_n(func, start, end):
    """parametres d'entree:
            func: liste de tuples pour decrire la fonction 
            (voir docstring de resolution_glouton)
            start: debut du domaine considere (pour les valeurs de m)
            end: fin
    on represente la courbe (valeurs de n) contre 
    (distance entre solution naive u_h et l'approximation Glouton)
    """
    xaxis = range(start, end)
    bla = Glouton(func, I=10, n=end, m=30).mem
    diff = np.array(bla[start:]) - resolution_naif(func)
    eps = list(map(lambda v: norm(v), diff))
    plt.plot(xaxis, eps)
    plt.xlabel("n")
    plt.ylabel("Erreur en n")
    plt.title(f"Convergence en n, P={len(func)}")
    plt.show()
    


## commenter/decommenter pour tester

#test_convergence_m(f1, 20, 130)
#test_convergence_m(f2, 20, 100)
#test_convergence_n(f1, 5, 50)
#test_convergence_n(f2, 5, 50)