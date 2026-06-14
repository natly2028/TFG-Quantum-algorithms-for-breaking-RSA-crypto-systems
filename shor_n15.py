# This code requires Qiskit and Qiskit Aer to be installed.

# Import the qiskit library
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit_ibm_runtime import Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import QFT
import matplotlib.pyplot as plt
from IPython.display import display
import math
import random
import numpy as np
from fractions import Fraction
%matplotlib inline



def classical_shor_pre_15():
    N=15
    #Check if N is a valid prime number
    if N%2==0:
        print("N is even")
        return (2, N//2), None      
    #Check if N is a perfect power
    for beta in range(2, int(math.log2(N)) + 1):
        alpha = round(N ** (1 / beta))
        if alpha ** beta == N:
            print(f"N is a perfect power: {alpha}^{beta}")
            return (alpha, N // alpha), None   
    #Choose a
    choice = input("Want to manually choose a? (Y/N): ").upper()
    if choice == 'Y':
        a = int(input(f"Introduce a value of a for N=15 such that 1 < a < {N}: "))
    else:
        a = random.randint(2, N - 1)
        print(f"a was randomly choosen to be a = {a}")

    g = math.gcd(a, N)

    if g > 1:
        print(f"gcd({a}, {N}) = {g}")
        return (g, N // g), None

    print("gcd(a, N) = 1 → order finding=quantumn shor")
    
    return None, (a,N)  # None indica que seguimos


def controlledU (a, power,ntrab,N):
    g = math.gcd(a, N)
    if g!=1:
        raise ValueError("a debe ser coprimo con N")
    U=QuantumCircuit(ntrab)
    for i in range (power):
        if a in [2, 13]:
            U.swap(2, 3)
            U.swap(1, 2)
            U.swap(0, 1)

        elif a in [7, 8]:
            U.swap(0, 1)
            U.swap(1, 2)
            U.swap(2, 3)
    
        elif a in [4, 11]:
            U.swap(0, 2)
            U.swap(1, 3)
            
    
        if a in [7, 11, 13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "{0}^{1} mod {2}".format(a, power, N)
    c_U = U.control()
    return c_U


def qft(n):
    qc = QuantumCircuit(n)
    for j in range(n-1, -1, -1):  
        qc.h(j)
        for k in range(j-1, -1, -1):
            qc.cp(np.pi / (2 ** (j - k)), k, j)

    for q in range(n // 2):
        qc.swap(q, n - q - 1)
    qc.name = "QFT"
    return qc


def shor_main():

    factors, params = classical_shor_pre_15()

    # Caso 1:clásicamente
    if factors is not None:
        p, q = factors
        print(f"Factorización encontrada sin cuántica: {p} * {q} = {p*q}")
        return factors, None, None, None, None

    # Caso 2:Shor cuántico
    else:
        a, N = params
        ncontrol = math.ceil(2 * math.log2(N))
        ntrab = math.ceil(math.log2(N))
        qc = QuantumCircuit(ncontrol + ntrab, ncontrol)
        qc.x(ncontrol)
        for q in range (ncontrol):
            qc.h(q)

        trab_qb = list(range(ncontrol, ncontrol + ntrab)) #Seleccionar qubits del registro de trabajo
        for q in range (ncontrol):
            power = 2**q
            qc.append(controlledU(a, power,ntrab, N), [q] + trab_qb)
            #qc.h(q)
        
        qc.append(qft(ncontrol).inverse(), range(ncontrol))
        
        for n in range(ncontrol):
            qc.measure(n, n)

        return None, qc, a, N, ncontrol


factors, qc, a, N, ncontrol = shor_main()

if factors is not None:
    p, q = factors
    print("Resultado final:", p, q)

else:
    print("Circuito listo para ejecutar")
    qc_fig = qc.draw("mpl", scale=0.45, fold=-1)
    qc_fig
    
    backend = AerSimulator()

    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    isa_qc = pm.run(qc)

    sampler = Sampler()
    job = sampler.run([isa_qc], shots=1024)
    result = job.result()

    counts = result[0].data.c.get_counts()
    print(counts)

    plot_histogram(counts, title=f"N = {N}, a = {a}", figsize=(6,6))
    plt.show()

    # candidate bitstrings ordered by probability (highest first), excluding all-zeros
    candidates = sorted(
        (k for k in counts if k != "0" * ncontrol),
        key=counts.get,
        reverse=True,
    )

    r = None
    measured_str = None
    p = q = None

    # walk down the list: most probable -> least probable,
    # accept the first bitstring whose r actually yields non-trivial factors
    for cand in candidates:
        s = int(cand, 2)
        frac = Fraction(s, 2 ** ncontrol).limit_denominator(N)
        r_candidate = frac.denominator

        # condition 1: r must be a real period -> a^r ≡ 1 (mod N)
        if pow(a, r_candidate, N) != 1:
            print(f"bitstring = {cand}, s = {s}, r = {r_candidate} "
                  f"-> incorrect value of r (a^r mod N != 1)")
            continue

        # condition 2: r must be even, so r/2 is an integer
        if r_candidate % 2 != 0:
            print(f"bitstring = {cand}, s = {s}, r = {r_candidate} "
                  f"-> incorrect value of r (r is odd)")
            continue

        # condition 3: a^(r/2) must NOT be -1 (mod N), else factors are trivial
        x = pow(a, r_candidate // 2, N)
        if x == N - 1:
            print(f"bitstring = {cand}, s = {s}, r = {r_candidate} "
                  f"-> incorrect value of r (a^(r/2) = -1 mod N)")
            continue

        # all conditions met -> extract the factors
        p_candidate = math.gcd(x - 1, N)
        q_candidate = math.gcd(x + 1, N)

        if p_candidate in (1, N) or q_candidate in (1, N):
            print(f"bitstring = {cand}, s = {s}, r = {r_candidate} "
                  f"-> incorrect value of r (trivial factors)")
            continue

        # valid candidate -> accept it and stop
        print(f"bitstring = {cand}, s = {s}, r = {r_candidate} -> correct value of r")
        r = r_candidate
        measured_str = cand
        p, q = p_candidate, q_candidate
        break

    if r is None:
        print("No bitstring gave a usable order r")
    else:
        print("bitstring =", measured_str)
        print("r final =", r)
        print("p =", p, "q =", q)
