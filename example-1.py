import numpy as np
import matplotlib.pyplot as plt
import math, cmath
from scipy.integrate import odeint
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from qiskit.quantum_info.operators import Operator, Pauli
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_aer import AerSimulator
from qiskit import transpile
from qiskit.circuit.library import UnitaryGate, HamiltonianGate

def rho_dot(rho, t, args, C):
    """
    Computes the time derivative of the density matrix
    coefficients in terms of the structure constants C
    of the base. This function implements Eq. (61).

    Output
    rho_dot: numpy array containing the elements of
        the time derivative of the density matrix
        coefficients in terms of the structure constants
        Q of the base

    Time derivative of the density matrix coefficients in
    terms of the structure constants Q of the base

    Parameters
    rho: vector (list or numpy array) containing the elements
        of the density matrix.
    t: time variable
    args: list containing the Hamiltonian parameters.
    C: numpy array containing the structure constants
        of the base.

    """
    a = ham_coefs(t, args)
    mat = np.transpose(np.tensordot(a, C, axes = ([0],[0])))
    rho_dot = np.matmul(mat, rho)

    return rho_dot

def classical_sol(dt, tdiv, rho0, args, C):
    """Obtains the classical solution for the coefficients of the
    density matrix numerically solving the time dependent
    von Neumann equation following the procedure outlined
    in the Supplemental Material.

    Output
    np.array(t_list): numpy array with the list of times used in the
        classical simulation.
    np.array(probs_list): numpy array with the list of probabilieties
        of the classical simulation corresponding to t_list.
    np.array(rho_list): numpy array with the list of density matrix
        coefficients corresponding to t_list.

    Parameters
    tmin: initial time in the classical simulation.
    tmax: final time in the classical simulation.
    tdiv: time divisions in the classical simulation.
    rho0: initial coefficients of the density matrix.
    args: list of the Hamiltonian arguments.
    C: numpy array containing the structure constants
        of the base.
    """

    #odeint

    probs_list = []
    rho_list = []
    t_list = []

    for k in range(tdiv):
        t = k * dt
        t_list.append(t)

    rhos = odeint(rho_dot, rho0, t_list, args=(args,C))

    for rho in rhos:
        probs = [amp ** 2 for amp in rho]
        probs_list.append(probs)
        rho_list.append(rho)

    return np.array(t_list), np.array(probs_list), np.array(rho_list)

def list_states(nq):
    """Lists the possible states in binary format of a
    register with nq qubits.

    Output
    states: list of strings in binary format containing the list
        of all possible states.

    Parameters
    nq: integer of the number of qubits.

    """
    states = []
    for k in range(2 ** nq):
        kbin = bin(k)[2:]
        while len(kbin) < nq:
            kbin = '0' + kbin
        states.append(kbin)

    return states

def Ui_gate(rho0, nq):
    """Calculates the U_i gate that initializes the second
    register to the density matrix elements.
    This function implements the initialization of
    the quantum circuit given by Eq. (67).

    Ouptut
    qc.to_gate(): Qiskit gate of the initialization unitary
        Householder matrix.

    Parameters
    rho0: initial coefficients of the density matrix.
    nq: integer number of the number of qubits.

    """

    dim = 2**(nq - 1)

    u = np.zeros(dim)
    u[0] = 1.0

    r0 = rho0 / np.sqrt(np.dot(rho0, rho0))

    w = np.array([r0 - u])

    #Householder matrix
    H = np.identity(dim) - np.matmul(np.transpose(w), w) / (1.0 - r0[0])

    qcH = QuantumCircuit(nq - 1)
    qcH.append(UnitaryGate(H), range(nq - 2, -1, -1))
    UH = qcH.to_gate(label='$U_H$').control(1)

    qc = QuantumCircuit(nq)

    qc.h(nq - 1)
    qc.append(UH, range(nq - 1, -1, -1))
    qc.x(nq - 1)

    for jj in range(nq - 1):
        qc.ch(nq - 1, jj)

    qc.x(nq - 1)

    return qc.to_gate(label='$U_I$')

def M_gate(alpha, nq, pauli_l):
    """Composes the M gate through a HamiltonianGate.
    This function implements the gate M in Fig. 3
    or Eq. (71)

    Output
    qM.to_gate(): gate of the M transformation

    Parameters
    alpha: numpy array containing the parameters of
        the M unitary transformation.
    nq: integer number of the number of qubits.
    pauli_l: list of lists of tuples containing the Pauli
       operator corresponding to the projection of C_k^{n}
       on h_i^{2n} given by Tr[C_k^{n}h_i^{2n}]h_i^{2n}
       and the corresponding coefficients Tr[C_k^{n}h_i^{2n}].
    """

    qM = QuantumCircuit(nq - 1)

    en_pauli_l = list(enumerate(pauli_l))
    en_pauli_l.reverse()

    for k, pau_l in en_pauli_l:
        for pau in pau_l:
            Hg = HamiltonianGate(pau[0], alpha[k] * pau[1])
            qM.append(Hg, range(nq - 2, -1 , -1))

    return qM.to_gate(label='M')

def qcircuit(dt, n, rho0, args, nq, pauli_l):
    """Builds the quantum circuit for n time steps of the
    M gate. This function implements the quantum circuit
    shown in Fig. 1.

    Output
    qcc: Qiskit QuantumCircuit containing a circuit of n
        time steps

    Parameters
    dt: real number of the size of the time step.
    n: integer number of time steps.
    args: list of the Hamiltonian arguments.
    rho0: initial coefficients of the density matrix.
    nq: integer number of the number of qubits.
    pauli_l: list of lists of tuples containing the Pauli
       operator corresponding to the projection of C_k^{n}
       on h_i^{2n} given by Tr[C_k^{n}h_i^{2n}]h_i^{2n}
       and the corresponding coefficients Tr[C_k^{n}h_i^{2n}]

    """

    qc = QuantumCircuit(nq - 1, name='$M({\\bf \\alpha})$')
    qcc = QuantumCircuit(nq, nq)
    Ui = Ui_gate(rho0, nq)
    qcc.append(Ui, range(nq))

    for k in range(n):
        dt_a = dt * ham_coefs(k * dt, args)

        qM = M_gate(dt_a, nq, pauli_l)

        qc.append(qM, range(nq - 1))

    #build controlled M gate
    cM_gate = qc.to_gate().control(1)

    qcc.append(cM_gate,  range(nq - 1, -1, -1))

    qcc.h(nq - 1)

    qcc.barrier(range(nq))

    qcc.measure(range(nq), range(nq))

    return qcc

def quantum_tomography(result, states, n0, nq, shots):
    """Does the quantum tomography of the counts.
    This function mainly uses Eqs. (78) and (79) to compute
    the density matrix coefficients.

    Output
    rho: list of density matrix coefficients.

    Parameter
    result: Qiskit Result of the circuit run
    states: list of strings in binary format containing the list
        of all possible states.
    n0: Normalization constant of the density matrix
    nq: integer with the number of qubits.
    shots: number of circuit executions.
    """
    count = result.get_counts()
    rho = []
    for state in states:
        p0 = count.get('0' + state, 0)/shots
        p1 = count.get('1' + state, 0)/shots
        a = n0 * np.sqrt(2**(nq - 1)) * (p0 - p1)
        rho.append(a)

    return rho

def pauli_base(nq):
    """Generates all the elements of an algebra given by
    (1/2)^{n/2}\sigma_{i_1} \otimes \sigma_{i_2} \otimes
    ... \sigma_{i_n} using Eq. (B1) iteratively where
    \sigma_0 is the 2 x 2 identity matrix and \sigma_1,
    \sigma_2 and \sigma_3 are the Pauli matrices

    Output
    base_1: numpy array containing all the elements of
        the matrix base.

    Parameters
    nq: integer with the number of qubits.

    """
    base_0 = np.array([
        (1.0/np.sqrt(2.0) + 0j) * np.identity(2),
        (1.0/np.sqrt(2.0) + 0j) * np.array([[0, 1], [1, 0]]),
        (1.0/np.sqrt(2.0) + 0j) * np.array([[0, -1j], [1j, 0]]),
        (1.0/np.sqrt(2.0) + 0j) * np.array([[1, 0], [0, -1]])
    ])

    base_1 = base_0[:]

    for k in range(nq - 1):
        base_2 = []
        for el1 in base_1:
            for el0 in base_0:
                base_2.append(np.kron(el1, el0))
        base_1 = np.array(base_2)

    return base_1

def structure_constants_base(base):
    """Calculates the structure constants of base.
    This function implements Eqs. (B10) and (B12).

    Output
    C: numpy array containing the structure constants
       of the commutator of base.
    B: numpy array containing the structure constants
       of the anticommutator of base.

    Parameters
    base: numpy array containing all the elements of
        the matrix base.

    """
    dim = np.shape(base)[0]
    C = 0j * np.zeros((dim, dim, dim))
    B = 0j * np.zeros((dim, dim, dim))
    for k0, el0 in enumerate(base):
        for k1, el1 in enumerate(base):
            commu = np.matmul(el0, el1) - np.matmul(el1, el0)
            anticommu = np.matmul(el0, el1) + np.matmul(el1, el0)
            for k2, el2 in enumerate(base):
                C[k0, k1, k2] = -1j * np.trace(np.matmul(commu, el2)) / np.trace(np.matmul(el2, el2))
                B[k0, k1, k2] = np.trace(np.matmul(anticommu, el2)) / np.trace(np.matmul(el2, el2))

    return [C, B]

def structure_constants_recursive(nq):
    """Recursively calculates the structure constants of base.
    This function implements Eqs. (B17) and (B18).

    Output
    Cn: numpy array containing the structure constants
        of the commutator of base.
    Bn: numpy array containing the structure constants
        of the anticommutator of base.s

    Parameters
    nq: ingeger with the number of qubits
    """
    base = pauli_base(1)

    C1, B1 = structure_constants_base(base)
    Cn = C1[:]
    Bn = B1[:]

    for _ in range(nq-1):
        Cnp = (np.kron(Bn, C1) + np.kron(Cn, B1))/2.0
        Bnp = (np.kron(Bn, B1) - np.kron(Cn, C1))/2.0
        Cn = Cnp[:]
        Bn = Bnp[:]

    return [Cn, Bn]

def ham_gate_coefs(nq):

    """Calculates the Hamiltonian gate coefficients
    Tr[C_k h_i^{(2)}] = Tr[C_k^{n}h_i^{2n}] recursively.
    This function implements the computation of such
    coefficients appearing in Eqs. (75), (B21) and (B22).

    Output
    Chn: numpy array of Tr[C_k^{nq}h_i^{2nq}]
    Bhn: numpy array of Tr[B_k^{nq}h_i^{2nq}]

    Parameters
    nq: integer with the number of qubits
    """

    h = pauli_base(2)
    C1, B1 = structure_constants_recursive(1)

    Ch1 = np.trace(np.tensordot(C1, h, axes = ([2,],[1,])), axis1 = 1, axis2 = 3)
    Bh1 = np.trace(np.tensordot(B1, h, axes = ([2,],[1,])), axis1 = 1, axis2 = 3)
    Chn = Ch1[:]
    Bhn = Bh1[:]

    for _ in range(nq-1):
        Chnp = (np.kron(Bhn, Ch1) + np.kron(Chn, Bh1))/2.0
        Bhnp = (np.kron(Bhn, Bh1) - np.kron(Chn, Ch1))/2.0
        Chn = Chnp[:]
        Bhn = Bhnp[:]

    return Chn, Bhn

def int_to_pauli(n, nq):
    """Converts an integer n in decimal base to
    the corresponding Pauli string in base four
    using the Qiskit standard notation I, X, Y and Z.
    Exampes: n = 0 -> 'I', n = 1 -> 'X',
             n = 21 -> 'XXX', n = 52 -> 'ZXI',
             n = 692 -> 'YYZXI'.

    Output
    stn: string with the Pauli string using the
         Qiskit standard notation I, X, Y and Z.

    Parameters
    n: integer in decimal base

    """
    res = n
    pau = ['I', 'X', 'Y', 'Z']
    stn = ''
    while res > 0:
        mo = res % 4
        stn = pau[mo] + stn
        res = res // 4

    while len(stn) < nq - 1:
        stn = 'I' + stn

    return(stn)

def pauli_list(nq):
    """Generates a list of Pauli operators and their corresponding
    coefficients (1/2)^{n/2}Tr[C_k^{n}h_i^{2n}] of Eq. (74) to be used
    by the Hamiltonian gates of Eq. (71) .

    Output
    pauli_list: list of lists of tuples containing the Pauli
                operator corresponding to the projection of C_k^{n}
                on h_i^{2n} given by Tr[C_k^{n}h_i^{2n}]h_i^{2n}
                and the corresponding coefficients Tr[C_k^{n}h_i^{2n}]
    Parameters
    nq: integer with the number of qubits
    """
    Chn, _ = ham_gate_coefs((nq - 1) // 2)
    pauli_list = []
    for chn in Chn:
        line = []
        for i, ch in enumerate(chn):
            if np.abs(ch) > 0:
                line.append((Pauli(int_to_pauli(i, nq)), np.imag(ch)/np.sqrt(2**(nq - 1))))
        pauli_list.append(line)

    return pauli_list

#############################################
# EXAMPLE 1                                 #
# One spin 1/2 particle                     #
# subject to an oscillating magnetic field. #
#############################################

def ham_coefs(t, args):
    """Calculates the coefficients for the nuclear
    resonance Hamiltonian of a spin 1/2 particle.
    
    Output
    ham: numpy array containing the Hamiltonian coefficients
    
    Parameters
    t: time variable.
    args: Hamiltonian arguments.
        omega: Angular frequency of the electromagnetic
            drive.
        omega0: energy separation of the two level
            system.
        omega1: Rabi frequency
    
    """

    
    dim, omega0, omega1, omega, phi = args
    ham = np.zeros(dim)
    sqt2 = np.sqrt(2.0)
    ham[1] = omega1 * np.cos(omega * t) / sqt2
    ham[2] = -omega1 * np.cos(omega * t + phi) / sqt2
    ham[3] = -omega0 / sqt2
    
    return ham

# Calculation of the structure constants
# of the Pauli base and the list of Pauli
# projections over C_k
print('base and structure constants')
dim = 4
nq = 3

base = pauli_base(1)
sC, _ = structure_constants_recursive(1)
C = np.real(sC)

# Values of the
# Hamiltonian parameters

omega0 = 1.0
omega1 = 22.0
omega = 0.90
phi = np.pi / 2

tmin = 0.0
tmax = 2 * np.pi / omega1

args = (dim, omega0, omega1, omega, phi)

# Initial conditions of the 
# density matrix coefficients
# expressed in terms of base

rho0 = [1 / np.sqrt(2.0), 0.0, 0.0, - 1 / np.sqrt(2.0)]

# Classical algorithm solution
tdiv = 500

dt = (tmax - tmin) / tdiv

print('classical solution')
t_list_od, probs_list_od, rho_list_od = classical_sol(dt,
                                            tdiv, rho0, args, C)
# Create a DataFrame for the classical
# solution and save it
print('saving classical results')
df_od = pd.DataFrame(np.transpose(np.append([t_list_od], np.transpose(rho_list_od), axis=0)))
df_od.to_csv('Results/example-1/results_od.csv')

# Setting the parameters
# for the quantum gates
print(' ')
print('quantum solution')

# Set backend
print('creating backend')
backend = AerSimulator()
exc = ThreadPoolExecutor(max_workers=100)
backend.set_options(executor=exc)
backend.set_options(max_job_size=1)

pauli_l = pauli_list(nq)
states = list_states(nq - 1)
n0 = np.sqrt(np.dot(rho0, rho0))

t_list_qg = []
rho_list_qg = []
max_div = 50
ndat = 1
for tdiv in range(1, max_div + 1):

    print('*****************************')
    print(tdiv, '/', max_div)
    tmin = 0.0
    tmax = tdiv / max_div * 2 * np.pi / omega1

    dt = (tmax - tmin) / tdiv

    # Build and transpile quantum circuits for the whole run
    # using the qasm simulator

    print('generating quantum circuit')
    qc = qcircuit(dt, tdiv, rho0, args, nq, pauli_l)

    # Transpile circuits
    print('transpiling circuit')
    tran_qc = transpile(qc, backend=backend)
    shots = 2048 * 16

    # Run the quantum circuit and retrieve results
    print('runing circuit')
    result = backend.run(tran_qc, shots=shots).result()

    # Quantum tomography
    # Calculate the list of probabilities
    # and the list of the density matrix
    # coeficients
    rho = quantum_tomography(result, states, n0, nq, shots)
    rho_list_qg.append(rho)
    t_list_qg.append(tdiv * dt)
    
    # Create a DataFrame for the quantum
    # solution and save it
    print('saving quantum solution')
    df_qg = pd.DataFrame(np.transpose(np.append([t_list_qg], np.transpose(rho_list_qg), axis=0)))
    df_qg.to_csv('Results/example-1/results_qg.csv')
