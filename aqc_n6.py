import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg
from scipy.sparse.linalg import spsolve
from scipy.special import factorial



# Definition of operators
Nx = 15
Ny = 15
# single h.o. operators
des = [np.sqrt(i) for i in range(1, Nx)]
a = sp.diags(des,1)
ad = np.transpose(a)
num = [i for i in range(Nx)]
n = sp.diags(num)
# full system operators
# x coordinate
ax = sp.kron(a, sp.eye(Nx))
adx = sp.kron(ad, sp.eye(Nx))
nx = sp.kron(n, sp.eye(Nx))
# y coordinate
ay = sp.kron(sp.eye(Nx), a)
ady = sp.kron(sp.eye(Nx), ad)
ny = sp.kron(sp.eye(Ny), n)



# initial Hamiltonian
def H0(tx, ty):
    return np.dot( (adx-np.conj(tx)*sp.eye(Nx*Ny)), (ax-tx*sp.eye(Nx*Ny)) ) +\
                np.dot( (ady-np.conj(ty)*sp.eye(Nx*Ny)), (ay-ty*sp.eye(Nx*Ny)) )

# final/problem Hamiltonian
def Hp(N):
    h1f = N*sp.eye(Nx*Ny)-np.dot(nx,ny)
    h2f = nx-ny
    h3f = np.dot(h2f, h2f)
    return N**2*np.dot(h1f,h1f)+np.dot(nx, h3f)

# time dependent Hmailtonian
def H(s, tx, ty, N):
    return (1-s)*H0(tx, ty)+s*Hp(N)

# initial state sorting
def lowesteig(Ham, neig=6):
    x, v = sp.linalg.eigs(Ham, k=neig, which='SR')
    idx = np.argsort(x.real)
    return x[idx], v[:, idx]



# Solve the time dependent Schrodinger equation assuming that the adiabatic approximation is fulfilled and store the 6 lowest eigestates energies.
points = 100
levels = 6
N = 6
thX = N**(1./4.)
thY = N**(1./4.)
s = np.linspace(0, 1, points)
energy = np.zeros((levels, points))
for i in range(points):
    energy[:,i], _ = lowesteig(H(s[i], thX, thY, N), neig=levels)



# Plot the previous instantaneous eigenenergies as a function of time.
colors = ['steelblue', 'darkolivegreen', 'mediumpurple',
          'lightcoral', 'lightseagreen', 'navy']

s_energy = np.linspace(0, 1, energy.shape[1])

for i in range(levels):
    plt.plot(s_energy,
             energy[i,:],
             color=colors[i],
             linewidth=2)

plt.ylim(0, 30)
plt.xlabel(r'$t/T$', fontsize=15)
plt.ylabel(r'Energy', fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.grid(alpha=0.3)

plt.show()



# From the final ground and excited states determine the prime factors nx and ny.

[x, v] = lowesteig(H(s[-1], thX, thY, N), neig=2)
factor0 = np.real(np.array([np.conj(v[:,0])@nx@v[:,0], np.conj(v[:,0])@ny@v[:,0]]))
factor1 = np.real(np.array([np.conj(v[:,1])@nx@v[:,1], np.conj(v[:,1])@ny@v[:,1]]))

print(f"*** Ground state non-trivial factors found: {factor0[0]:.0f} and {factor0[1]:.0f} ***")   
print(f"*** Excited state non-trivial factors found: {factor1[0]:.0f} and {factor1[1]:.0f} ***") 



# Initial coherent state

def coherent_state(alpha, dim):
    coeffs = np.array([
        np.exp(-abs(alpha)**2/2) * alpha**n / np.sqrt(factorial(n))
        for n in range(dim)
    ], dtype=complex)
    return coeffs / np.linalg.norm(coeffs)

psi_x = coherent_state(thX, Nx)
psi_y = coherent_state(thY, Ny)

psi0 = np.kron(psi_x, psi_y)
psi0 = psi0 / np.linalg.norm(psi0)

# Basis states

def basis_index(nx_val, ny_val):
    return nx_val * Ny + ny_val

idx_23 = basis_index(2, 3)
idx_32 = basis_index(3, 2)

# Matrices

H0_mat = sp.csr_matrix(H0(thX, thY))
Hp_mat = sp.csr_matrix(Hp(N))
I_mat = sp.eye(Nx * Ny, format="csr")

# T values

Tvals = np.array([1, 5, 10, 20, 50])
steps = 80
ds = 1 / steps

prob_23 = np.zeros(len(Tvals))
prob_32 = np.zeros(len(Tvals))

# Evolution

for j, T in enumerate(Tvals):

    psi = psi0.copy()

    for i in range(steps):

        sval = (i + 0.5) / steps

        Ham = (1 - sval) * H0_mat + sval * Hp_mat

        A = I_mat + 1j * T * ds * Ham / 2
        B = I_mat - 1j * T * ds * Ham / 2

        psi = spsolve(A, B @ psi)
        psi = psi / np.linalg.norm(psi)

    prob_23[j] = np.abs(psi[idx_23])**2
    prob_32[j] = np.abs(psi[idx_32])**2

print("P(2,3) =", prob_23)
print("P(3,2) =", prob_32)



# Relative probability histogram: P(2,3) and P(3,2) normalized between them

total = prob_23 + prob_32

prob_23_rel = prob_23 / total
prob_32_rel = prob_32 / total

x = np.arange(len(Tvals))
width = 0.35

plt.bar(x - width/2, prob_23_rel, width, label=r'$|2,3\rangle$', color='teal')
plt.bar(x + width/2, prob_32_rel, width, label=r'$|3,2\rangle$', color='paleturquoise')

plt.xlabel(r'$T$', fontsize=15)
plt.ylabel('Relative probability', fontsize=15)

plt.xticks(x, Tvals, fontsize=13)
plt.yticks(fontsize=13)

plt.ylim(0, 1)

plt.legend(fontsize=13)
plt.grid(True)

plt.show()
