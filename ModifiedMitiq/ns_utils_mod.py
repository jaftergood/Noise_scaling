from qiskit import QuantumCircuit, transpile
from mitiq import QPROGRAM
import numpy as np
from typing import cast, List, Optional
import numpy.typing as npt
# from mitiq.pec.types import OperationRepresentation
from mitiq.pec.representations.optimal import minimize_one_norm
from qutip import sigmax, sigmay, sigmaz, sigmam, sigmap, qeye
from ModifiedMitiq.NoisyOperationMod import NoisyOperation
from ModifiedMitiq.OperationRepresentationMod import OperationRepresentation


def _Trotter(
             dtJ: float,
             N: int, 
             backend=None,
             trots: int = 1,
             TFIM: bool = False,
             dtH: float = None,
             x: bool = True,
             xx: bool = True,
             pbc: bool = True,
) -> QuantumCircuit:
    '''
    A Trotter function for the Ising model. Makes trots Trotter steps of
    step size dt. Cab put circuit in x basis to begin with, and measures in
    x basis as well.
    
    INPUTS:
    
    dtJ :: Trotter step size (multiplied by ZZ interaction strength).
    N :: The number of qubits.
    backend :: Can supply a backend to transpile with, if desired.
    trots :: The number of Trotter steps to take; default is 1.
    TFIM :: Controls whether to use TFIM or not.
    dtH :: Trotter step size (multiplied by X interaction strength).
    x :: Boolean, default True, that initializes x basis.
    xx :: Boolean, default True, that measures in x basis.
    pbc :: Boolean, default True, that governs periodic boundary conditions.
    
    RETURNS:
    
    The Trotter circuit in Qiskit form.
    '''
    xyz_circuit = QuantumCircuit(N)
    if x:
        qc_init = QuantumCircuit(N)
        qc_init.h(range(N))
        qc_init.barrier()
    for _ in range(trots):
        for n in range(0, N-1, 2):
            xyz_circuit.rzz(dtJ, n, n+1)
        for n in range(1, N-1, 2):
            xyz_circuit.rzz(dtJ, n, n+1)
        if pbc:
            xyz_circuit.rzz(dtJ, N-1, 0)
        if TFIM and dtH is not None:
            for n in range(N):
                xyz_circuit.rx(dtH, n)
        if TFIM and dtH is None:
            raise ValueError('The transverse field must have a value.')
        xyz_circuit.barrier()
    if xx:
        qc_meas = QuantumCircuit(N)
        qc_meas.h(range(N))
        
    if backend is not None:
        xyz_circuit_t = transpile(xyz_circuit, backend, optimization_level=0)
    
    if x and xx:
        circuit_out = (qc_init.compose(
            xyz_circuit_t.decompose().decompose().decompose().decompose())
                      ).compose(qc_meas)
    if x and not xx:
        circuit_out = (qc_init.compose(
            xyz_circuit_t.decompose().decompose().decompose().decompose())
                      )
    if not x and xx:
        circuit_out = (
            xyz_circuit_t.decompose().decompose().decompose().decompose()
                      ).compose(qc_meas)
    if not x and not xx:
        circuit_out = xyz_circuit_t.decompose().decompose().decompose().decompose()
    
    return circuit_out

def find_optimal_representation_super(
    ideal_operation: QPROGRAM,
    noisy_super_operator: npt.NDArray[np.complex64],
    noisy_operations: List[NoisyOperation],
    tol: float = 1.0e-8,
    initial_guess: Optional[npt.NDArray[np.float64]] = None,
) -> OperationRepresentation:
    r"""Returns the ``OperationRepresentaiton`` of the input ideal operation
    which minimizes the one-norm of the associated quasi-probability
    distribution.

    More precicely, it solves the following optimization problem:

    .. math::
        \min_{{\eta_\alpha}} = \sum_\alpha |\eta_\alpha|,
        \text{ such that }
        \mathcal G = \sum_\alpha \eta_\alpha \mathcal O_\alpha,

    where :math:`\{\mathcal O_j\}` is the input basis of noisy operations.

    Args:
        ideal_operation: The circuit element to approximate, as a QPROGRAM.
        noisy_super_operator: The noisy super operator to approximate; i.e.,
            the (A) circuit to approximate in terms of (B) circuits, and input
            as a npt.NDArray[np.complex64]
        noisy_basis: The ``NoisyBasis`` in which the ``ideal_operation``
            should be represented. It must contain ``NoisyOperation`` objects
            which are initialized with a numerical superoperator matrix.
        tol: The error tolerance for each matrix element
            of the represented operation.
        initial_guess: Optional initial guess for the coefficients
            :math:`\{ \eta_\alpha \}``.

    Returns: The optimal OperationRepresentation.
    """
    # ideal_cirq_circuit, _ = convert_to_mitiq(ideal_operation)
    # ideal_matrix = kraus_to_super(
    #     cast(List[npt.NDArray[np.complex64]], kraus(ideal_cirq_circuit))
    # )
    ideal_matrix = noisy_super_operator
    
    try:
        basis_matrices = [
            noisy_op.channel_matrix for noisy_op in noisy_operations
        ]
    except ValueError as err:
        if str(err) == "The channel matrix is unknown.":
            raise ValueError(
                "The input noisy_basis should contain NoisyOperation objects"
                " which are initialized with a numerical superoperator matrix."
            )
        else:
            raise err  # pragma no cover
    
    # Run numerical optimization problem
    quasi_prob_dist = minimize_one_norm(
        ideal_matrix,
        basis_matrices,
        tol=tol,
        initial_guess=initial_guess,
    )
    
    # basis_expansion = {op: eta for op, eta in zip(basis_set, quasi_prob_dist)}
    return OperationRepresentation(
        ideal_operation, noisy_operations, quasi_prob_dist.tolist()
    )

def z_Trotter_new(
                  cnot: OperationRepresentation,
                  rz: OperationRepresentation,
                  N: int, 
                  backend,
                  trots: int = 1,
                  x: bool = True,
                  xx: bool = True,
                  h: OperationRepresentation = None,
                  pbc: bool = True,
                  TFIM: bool = False,
                  rx: OperationRepresentation = None,
                 ):
    '''
    A Trotter function. Give it the expansions for a CNOT and an rz gate
    and it will return a circuit of gates selected from the expansions according to
    the probabilities derived from the expansions. Makes trots Trotter steps (step
    size is determined when performing the expansion).
    
    INPUTS:
    
    cnot :: Object that contains the expansion and coefficients of the cnot operator.
    rz :: Object that contains the expansion and coefficients of the rz operator.
    N :: The number of qubits.
    backend :: The backend for transpilation.
    trots :: The number of Trotter steps to take; default is 1.
    x :: Boolean value (True by default) that initializes the circuit in the x basis.
    h :: None by default; give this parameter if x basis is desired. Needed for x and xx.
    xx :: Boolean value (True by default) that measures the circuit in the x basis.
    pbc :: Boolean, True by default, and gives the circuit periodic boundaries.
    TFIM :: Boolean, False by default; turns on TFIM when True.
    rx :: None by default; give this parameter if TFIM is desired.
    
    RETURNS:
    
    The Trotter circuit in Qiskit form, the total overhead of the circuit, and the final sign
    associated with a given sampled circuit.
    '''
    xyz_circuit = QuantumCircuit(N)
    cardinality = 1
    gamma = 1
    if x:
        qc_initialize = QuantumCircuit(N)
        if h is None:
            qc_initialize.h(range(N))
            if not xx:
                print('Note that the Hadamard gates may have full APD noise (check to be sure).')
        else:
            for n in range(N):
                tempHad = h.sample()
                gamma *= h.norm
                cardinality *= tempHad[1]
                qc_initialize.append(tempHad[0].native_circuit, [n])
    for _ in range(trots):
        xyz_circuit.barrier()
        for n in range(0, N-1, 2):
            gamma *= cnot.norm * rz.norm * cnot.norm
            temp0 = cnot.sample()
            temp1 = rz.sample()
            temp2 = cnot.sample()
            cardinality *= temp0[1] * temp1[1] * temp2[1]
            xyz_circuit.append(temp0[0].native_circuit, [n, n+1])
            xyz_circuit.append(temp1[0].native_circuit, [n+1])
            xyz_circuit.append(temp2[0].native_circuit, [n, n+1])
        for n in range(1, N-1, 2):
            gamma *= cnot.norm * rz.norm * cnot.norm
            temp0 = cnot.sample()
            temp1 = rz.sample()
            temp2 = cnot.sample()
            cardinality *= temp0[1] * temp1[1] * temp2[1]
            xyz_circuit.append(temp0[0].native_circuit, [n, n+1])
            xyz_circuit.append(temp1[0].native_circuit, [n+1])
            xyz_circuit.append(temp2[0].native_circuit, [n, n+1])
        if pbc:
            gamma *= cnot.norm * rz.norm * cnot.norm
            temp0 = cnot.sample()
            temp1 = rz.sample()
            temp2 = cnot.sample()
            cardinality *= temp0[1] * temp1[1] * temp2[1]
            xyz_circuit.append(temp0[0].native_circuit, [0, n+1])
            xyz_circuit.append(temp1[0].native_circuit, [n+1])
            xyz_circuit.append(temp2[0].native_circuit, [0, n+1])
        xyz_circuit.barrier()
        if TFIM and rx is not None:
            # xyz_circuit.barrier()
            for n in range(N):
                temp3 = rx.sample()
                gamma *= rx.norm
                cardinality *= temp3[1]
                xyz_circuit.append(temp3[0].native_circuit, [n])
        if TFIM and rx is None:
            raise ValueError('Requires the optimal representation of rx if TFIM desired.')
    xyz_circuit.barrier()
    if xx:
        qc_measure = QuantumCircuit(N)
        if h is None:
            qc_measure.h(range(N))
        else:
            for n in range(N):
                tempHad = h.sample()
                gamma *= h.norm
                cardinality *= tempHad[1]
                qc_measure.append(tempHad[0].native_circuit, [n])
        qc_measure.barrier()
    
    xyz_circuit_t = transpile(xyz_circuit, backend, optimization_level=0)
    
    if x and xx:
        final_circuit = (qc_initialize.compose(
                    xyz_circuit_t.decompose().decompose().decompose())
                        ).compose(qc_measure)
    if x and not xx:
        final_circuit = (qc_initialize.compose(
                    xyz_circuit_t.decompose().decompose().decompose())
                        )
    if not x and xx:
        final_circuit = (
                    xyz_circuit_t.decompose().decompose().decompose()
                        ).compose(qc_measure)
    if not x and not xx:
        final_circuit = xyz_circuit_t.decompose().decompose().decompose()
    
    return (final_circuit, gamma, cardinality)

def mk_circ(
    one: str = None,
    two: str = None,
    cx: str = None,
    theta: float = None,
    ):

    '''
    Make Qiskit circuits for this problem. Not meant to be general.
    '''
    
    if one is not None and two is None:
        qc = QuantumCircuit(1)
    if one is not None and two is not None and cx is None:
        qc = QuantumCircuit(2)
    if one is None and two is None and cx is not None:
        qc = QuantumCircuit(2)
    if one is None and two is None and cx is None:
        qc = QuantumCircuit(2)
    
    if one is not None:
        if one == 'i':
            qc.id(0)
        elif one == 'x':
            qc.x(0)
        elif one == 'y':
            qc.y(0)
        elif one == 'z':
            qc.z(0)
        elif one == 's':
            qc.sx(0)
        elif one == 'h':
            qc.h(0)
        elif one == 'rz':
            if theta is None:
                raise ValueError("theta can't be None")
            else:
                qc.rz(theta, 0)
        elif one == 'rx':
            if theta is None:
                raise ValueError("theta can't be None")
            else:
                qc.rx(theta, 0)
        else:
            raise ValueError("one must be one of 'i', 'x', 'y', 'z', 's', 'rx', 'rz'")
    
    if two is not None:
        if two == 'i':
            qc.id(1)
        elif two == 'x':
            qc.x(1)
        elif two == 'y':
            qc.y(1)
        elif two == 'z':
            qc.z(1)
        elif two == 's':
            qc.sx(1)
        elif two == 'h':
            qc.h(1)
        elif two == 'rz':
            if theta is None:
                raise ValueError("theta can't be None")
            else:
                qc.rz(theta, 1)
        elif two == 'rx':
            if theta is None:
                raise ValueError("theta can't be None")
            else:
                qc.rx(theta, 1)
        else:
            raise ValueError("two must be one of 'i', 'x', 'y', 'z', 's', 'rx', 'rz'")
    
    if cx is not None:
        qc.cnot(0, 1)
    
    return qc

def expand_an_operator(tot_num_sites, # Total number of sites
                       site_one, # Must be less than total number of sites
                       site_two, # Must be less than total number of sites
                       type_one=None, # Must be 'X', 'Y', 'Z', 'D', or 'U'
                       type_two=None, # Must be 'X', 'Y', 'Z', 'D', or 'U'
                      ):
    
    internal_dict = {'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz(),
                     'I': qeye(2),  'D': sigmam(), 'U': sigmap()}
    
    if type_one != None:
        a = internal_dict[type_one]
    else:
        a = qeye(2)
    
    if type_two != None:
        b = internal_dict[type_two]
    else:
        b = qeye(2)
        
    c = []
    for n in range(tot_num_sites):
        if n is site_one:
            c.append(a)
        elif n is site_two:
            c.append(b)
        else:
            c.append(qeye(2))
            
    return tensor(c)

def process_mag(
                N: int = 2,
                dt: float = 0.1,
                shots: int = 20000, 
                steps: int = 10, 
                backend = None,
                trotter_vars: tuple = None,
                ):
    if backend is None:
        backend = AerSimulator()
    if trotter_vars is None:
        xmeas, xbasis, TFIM = True, True, True
    else:
        xmeas, xbasis, TFIM = trotter_vars
    steps_ = np.linspace(0, int(steps), int(steps) + 1)
    trott_dicts = []
    for step in steps_:
        qc = _Trotter(dt, N, backend, int(step), x=xbasis, 
                        xx=xmeas, pbc=False, TFIM=TFIM, dtH=dt)
        qc.measure_all()

        res = backend.run(qc, shots=shots)
        trott_dicts.append(res.result().get_counts())
    prob_results = []
    for res_dict in trott_dicts:
        temp_dict = {}
        for i in range(N):
            temp_dict[f'x{i}u'] = 0
            for key in res_dict:
                if key[i] == '0':
                    temp_dict[f'x{i}u'] += res_dict[key]/shots
        prob_results.append(temp_dict)
    mx_dict = {}
    for step, prob in zip(steps_, prob_results):
        mx_dict[step * dt / 2] = 0
        for key in prob:
            mx_dict[step * dt / 2] += (2*prob[key] - 1)
    thymes = list(mx_dict.keys())
    mx_shots = list(mx_dict.values())
    return (thymes, mx_shots)

def z_Trotter_2rx(
                  rz1: OperationRepresentation,
                  N: int, 
                  backend,
                  trots: int = 1,
                  x: bool = True,
                  xx: bool = True,
                  pbc: bool = True,
                  TFIM: bool = False,
                  rx0: OperationRepresentation = None,
                  rx1: OperationRepresentation = None,
                 ):
    '''
    A Trotter function on 2 qubits. Give it the expansions for a rz gate on qb 1 and 
    an rx gate on qb 0 and 1 and it will return a circuit of gates selected from the 
    expansions according to the probabilities derived from the expansions. Makes trots 
    Trotter steps (step size is determined when performing the expansion).
    
    INPUTS:
    
    rz1 :: Object that contains the expansion and coefficients of the rz operator on qb1.
    N :: The number of qubits.
    backend :: The backend for transpilation.
    trots :: The number of Trotter steps to take; default is 1.
    x :: Boolean value (True by default) that initializes the circuit in the x basis.
    xx :: Boolean value (True by default) that measures the circuit in the x basis.
    pbc :: Boolean, True by default, and gives the circuit periodic boundaries.
    TFIM :: Boolean, False by default; turns on TFIM when True.
    rx0 :: None by default; give this parameter if TFIM is desired on qb0.
    rx1 :: None by default; give this parameter if TFIM is desired on qb1.
    
    RETURNS:
    
    The Trotter circuit in Qiskit form, the total overhead of the circuit, and the final sign
    associated with a given sampled circuit.
    '''
    xyz_circuit = QuantumCircuit(N)
    cardinality = 1
    gamma = 1
    if x:
        qc_initialize = QuantumCircuit(N)
        qc_initialize.h(range(N))
    for _ in range(trots):
        xyz_circuit.barrier()
        for n in range(0, N-1, 2):
            gamma *= rz1.norm
            temp = rz1.sample()
            cardinality *= temp[1]
            xyz_circuit.cx(n, n+1)
            xyz_circuit.append(temp[0].native_circuit, [n+1])
            xyz_circuit.cx(n, n+1)
        for n in range(1, N-1, 2):
            gamma *= rz1.norm
            temp = rz1.sample()
            cardinality *= temp[1]
            xyz_circuit.cx(n, n+1)
            xyz_circuit.append(temp[0].native_circuit, [n+1])
            xyz_circuit.cx(n, n+1)
        if pbc:
            gamma *= rz1.norm
            temp = rz1.sample()
            cardinality *= temp[1]
            xyz_circuit.cx(0, n+1)
            xyz_circuit.append(temp[0].native_circuit, [n+1])
            xyz_circuit.cx(0, n+1)
        xyz_circuit.barrier()
        if TFIM and rx0 is not None and rx1 is not None:
            # xyz_circuit.barrier()
            temp1 = rx0.sample()
            temp2 = rx1.sample()
            gamma *= rx0.norm * rx1.norm
            cardinality *= temp1[1] * temp2[1]
            xyz_circuit.append(temp1[0].native_circuit, [0])
            xyz_circuit.append(temp2[0].native_circuit, [1])
        if TFIM and (rx0 is None or rx1 is None):
            raise ValueError('Requires the optimal representation of both rx0 and rx1 if TFIM desired.')
    xyz_circuit.barrier()
    if xx:
        qc_measure = QuantumCircuit(N)
        qc_measure.h(range(N))
        qc_measure.barrier()
    
    xyz_circuit_t = transpile(xyz_circuit, backend, optimization_level=0)
    
    if x and xx:
        final_circuit = (qc_initialize.compose(
                    xyz_circuit_t.decompose().decompose().decompose())
                        ).compose(qc_measure)
    if x and not xx:
        final_circuit = (qc_initialize.compose(
                    xyz_circuit_t.decompose().decompose().decompose())
                        )
    if not x and xx:
        final_circuit = (
                    xyz_circuit_t.decompose().decompose().decompose()
                        ).compose(qc_measure)
    if not x and not xx:
        final_circuit = xyz_circuit_t.decompose().decompose().decompose()
    
    return (final_circuit, gamma, cardinality)
