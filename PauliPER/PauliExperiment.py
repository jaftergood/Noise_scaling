import numpy as np
from typing import List

from qiskit import QuantumCircuit
from qiskit.quantum_info import SuperOp

from qutip import Qobj, vector_to_operator

from PauliPER.PauliNoise import *
from PauliPER.ns_utils_ import mk_circ


def get_gates(labels: List[str], probabilities: List[float], base_values: List[float]):
    return [(np.random.choice(['II', a], p=[b, 1-b]), c) for a, b, c in zip(labels, probabilities, base_values)]

class PauliExperiment:
    
    def __init__(self, noise_object:PauliNoise, trotter_time_step:float, Jz:float=1, hx:float=1):
        self.noise_object = noise_object
        self.gates = self.hamiltonian(trotter_time_step, Jz, hx)
        self.gate_super_ops = []
        self.results_dict = dict()
        self.histogram = None
        self.density_matrices = dict()
        self.final_dm = None
        
    def __str__(self):
        return f"{dict(self.__iter__())}"

    def __iter__(self):
        yield 'hamiltonian_gate', self.gates
        yield 'results', self.results_dict
        yield 'density_matrices', self.density_matrices
        yield 'final_density_matrix', self.final_dm

    def hamiltonian(self, trotter_time_step:float, Jz:float, hx:float):
        ''' Populates the Hamiltonian circuits. '''
        return (mk_circ(cx='cx'), mk_circ('i','rz', theta=Jz*trotter_time_step), mk_circ('rx','i', theta=hx*trotter_time_step),
                      mk_circ('i', 'rx', theta=hx*trotter_time_step))
    
    def make_gate_super_ops(self):
        dim_list = [[2,2] for _ in range(self.noise_object.num_qubits)]
        self.gate_super_ops = tuple(Qobj(SuperOp(gate).data, dims=[dim_list, dim_list]) for gate in self.gates)

    def evolve_trotter_tfim(self, trotter_steps:int):
        if len(self.gate_super_ops) > 0:
            cx, rz1, rx0, rx1 = self.gate_super_ops
        else:
            self.make_gate_super_ops()
            cx, rz1, rx0, rx1 = self.gate_super_ops
        self.trotter_steps = int(trotter_steps)
        trotter_gate = 1; total_gamma = 1; num_pauli = 0
        for _ in range(int(trotter_steps)):
            cx1_err_gate = 1; cx2_err_gate = 1; rz1_err_gate = 1; rx0_err_gate = 1; rx1_err_gate = 1
            cx1_error = get_gates(self.noise_object.labels, self.noise_object.inverse_omegas, self.noise_object.inverse)
            cx2_error = get_gates(self.noise_object.labels, self.noise_object.inverse_omegas, self.noise_object.inverse)
            rz1_error = get_gates(self.noise_object.labels[3:6], self.noise_object.inverse_omegas[3:6], self.noise_object.inverse[3:6])
            rx0_error = get_gates(self.noise_object.labels[0:3], self.noise_object.inverse_omegas[0:3], self.noise_object.inverse[0:3])
            rx1_error = get_gates(self.noise_object.labels[3:6], self.noise_object.inverse_omegas[3:6], self.noise_object.inverse[3:6])
            for ele in cx1_error:
                if ele[0] != 'II':
                    cx1_err_gate *= self.noise_object.expanded_gates[ele[0]] * cx1_err_gate
                    if ele[1] > 0:
                        num_pauli += 1
            total_gamma *= self.noise_object.gammas[2]
            for ele in cx2_error:
                if ele[0] != 'II':
                    cx2_err_gate *= self.noise_object.expanded_gates[ele[0]] * cx2_err_gate
                    if ele[1] > 0:
                        num_pauli += 1
            total_gamma *= self.noise_object.gammas[2]
            for ele in rz1_error:
                if ele[0] != 'II':
                    rz1_err_gate *= self.noise_object.expanded_gates[ele[0]] * rz1_err_gate
                    if ele[1] > 0:
                        num_pauli += 1
            total_gamma *= self.noise_object.gammas[1]
            for ele in rx0_error:
                if ele[0] != 'II':
                    rx0_err_gate *= self.noise_object.expanded_gates[ele[0]] * rx0_err_gate
                    if ele[1] > 0:
                        num_pauli += 1
            total_gamma *= self.noise_object.gammas[0]
            for ele in rx1_error:
                if ele[0] != 'II':
                    rx1_err_gate *= self.noise_object.expanded_gates[ele[0]] * rx1_err_gate
                    if ele[1] > 0:
                        num_pauli += 1
            total_gamma *= self.noise_object.gammas[1]

            trotter_gate = (
                            rx1_err_gate * self.noise_object.super_ops_lam[1] * rx1 * 
                            rx0_err_gate * self.noise_object.super_ops_lam[0] * rx0 * 
                            cx2_err_gate * self.noise_object.super_ops_lam[2] * cx *
                            rz1_err_gate * self.noise_object.super_ops_lam[1] * rz1 * 
                            cx1_err_gate * self.noise_object.super_ops_lam[2] * cx *
                            trotter_gate
                            )
        
        return trotter_gate, num_pauli, total_gamma
    
    def run_experiment(self, reps:int, trotter_steps:int, rand_seed:int=None):
        if rand_seed is not None:
            np.random.seed(rand_seed)
        for rep in range(reps):
            self.results_dict[f'{rep}'] = self.evolve_trotter_tfim(trotter_steps)
            
    def process_density_matrices(self, initial_state:Qobj):
        '''
        initial_state must be in operator-ket form via QuTiP. That is: make a density 
        matrix and run operator_to_vector on it.
        '''
        if len(self.results_dict) < 1:
            raise ValueError("Must run the run_experiment method to get data first.")
        for key in self.results_dict:
            self.density_matrices[key] = (vector_to_operator(self.results_dict[key][0] * initial_state),
                                          self.results_dict[key][1], self.results_dict[key][2])
    
    def create_final_dm(self):
        final_dm = 0
        for key in self.density_matrices:
            holder = self.density_matrices[key]
            final_dm += (holder[2]*((-1)**(holder[1]))*holder[0])/len(self.density_matrices)
        self.final_dm = final_dm

    def evolve_initial_without_mitigation(self, initial_state:Qobj, trotter_steps:int=None):
        ''' initial_state must be in operator-ket form via QuTiP. '''
        cx, rz1, rx0, rx1 = self.gate_super_ops
        trotter_gate = 1
        if trotter_steps is None:
            trotter_steps = self.trotter_steps
        for step in range(int(trotter_steps)):
            trotter_gate = (
                            self.noise_object.super_ops_lam[1] * rx1 * 
                            self.noise_object.super_ops_lam[0] * rx0 * 
                            self.noise_object.super_ops_lam[2] * cx *
                            self.noise_object.super_ops_lam[1] * rz1 * 
                            self.noise_object.super_ops_lam[2] * cx *
                            trotter_gate
                            )
        return vector_to_operator(trotter_gate * initial_state)

    def evolve_target_without_mitigation(self, initial_state:Qobj, trotter_steps:int=None):
        ''' initial_state must be in operator-ket form via QuTiP. '''
        cx, rz1, rx0, rx1 = self.gate_super_ops
        trotter_gate = 1
        if trotter_steps is None:
            trotter_steps = self.trotter_steps
        for step in range(int(trotter_steps)):
            trotter_gate = (
                            self.noise_object.super_ops_targ[1] * rx1 * 
                            self.noise_object.super_ops_targ[0] * rx0 * 
                            self.noise_object.super_ops_targ[2] * cx *
                            self.noise_object.super_ops_targ[1] * rz1 * 
                            self.noise_object.super_ops_targ[2] * cx *
                            trotter_gate
                            )
        return vector_to_operator(trotter_gate * initial_state)


