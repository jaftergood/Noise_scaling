import numpy as np
from typing import List

from qiskit import QuantumCircuit
from qiskit.quantum_info import SuperOp

from qutip import qeye, sigmax, sigmay, sigmaz, tensor, Qobj, vector_to_operator

from mitiq.pec.channels import kraus_to_super

import matplotlib.ticker as tck
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def model_coeffs(
    base1: float = 1e-3,
    base2: float = 5e-4,
    zz: bool = False,
    m: float = 2.,
    rnd_seed: int = None
                ):
    
    if rnd_seed is not None:
        np.random.seed(rnd_seed)

    delta1 = np.linspace(-0.75*base1, 0.75*base1, 51)
    delta2 = np.linspace(-0.5*base2, 0.5*base2, 51)
    single_qubit_errors = [base1 + np.random.choice(delta1) for _ in range(6)]
    two_qubit_errors = [base2 + np.random.choice(delta2) for _ in range(9)]
    model_coeffs_lambda = single_qubit_errors + two_qubit_errors
    if zz:
        model_coeffs_phi = [model_coeffs_lambda[i]/m if i in [2, 5, 14] else 0 for i in
                            range(len(model_coeffs_lambda))]
    else:
        model_coeffs_phi = [base1 if n < 6 else base2 for n in range(len(model_coeffs_lambda))]

    return model_coeffs_lambda, model_coeffs_phi

def mk_omega(data: list):
    return [0.5 * (1 + np.exp(-2 * np.abs(x))) for x in data]


class PauliNoise:
    
    def __init__(self, lambdas, labels):
        self.base = {'I': qeye(2), 'X': sigmax(), 'Y': sigmay(), 'Z': sigmaz()}
        self.lambdas = lambdas
        self.labels = labels
        self.num_qubits = len(self.labels[0])
        self.lam_omegas = mk_omega(self.lambdas) # [0.5*(1 + np.exp(-2 * x)) for x in self.lambdas]
        self.model_gates, self.expanded_gates = self.populate_gate_lists()
        self.kraus_ops_lam = self.generate_kraus_ops()
        self.super_ops_lam = self.generate_super_ops()
        self.target = None
        self.target_omegas = None
        self.kraus_ops_targ = None
        self.super_ops_targ = None
        self.inverse = None
        self.inverse_omegas = None
        self.gammas = None
        
    def __str__(self):
        return f"{dict(self.__iter__())}" # "This class constructs the Pauli noise models (initial and target), and the inverse mapping between them."

    def __iter__(self):
        yield 'base', self.base
        yield 'lambdas', self.lambdas
        yield 'labels', self.labels
        yield 'num_qubits', self.num_qubits
        yield 'lam_omegas', self.lam_omegas
        yield 'model_gates', self.model_gates
        yield 'expanded_gates', self.expanded_gates
        yield 'kraus_ops_lam', self.kraus_ops_lam
        yield 'super_ops_lam', self.super_ops_lam
        yield 'target', self.target
        yield 'target_omegas', self.target_omegas
        yield 'kraus_ops_targ', self.kraus_ops_targ
        yield 'super_ops_targ', self.super_ops_targ
        yield 'inverse', self.inverse
        yield 'inverse_omegas', self.inverse_omegas
        yield 'overhead', self.gammas

        
    def populate_gate_lists(self):
        ''' This one is general. Extends to > 2 qubits. '''
        tensor_lists = [[self.base[el] for el in ele] for ele in self.labels]
        dim_list = [[2,2] for _ in range(self.num_qubits)]
        return ({key: Qobj(tensor(ele), dims=dim_list) 
                          for key, ele in zip(self.labels, tensor_lists)},
                {key: Qobj(tensor(tensor(ele), tensor(ele).trans()), dims=[dim_list, dim_list])
                          for key, ele in zip(self.labels, tensor_lists)})
    
    def generate_kraus_ops(self, diff_omegas: List[float] = None):
        ''' 
        This is not yet general. Only works for n=2 qubits. In particular,
        the probabilities need to be in order: 0:3 are qubit 0, 3:6 are qubit 1,
        and then the whole list applies to the 2-qubit error.
        '''
        if diff_omegas is None:
            comb_list = [(w, x) for w, x in zip(self.lam_omegas, self.labels)]
        else:
            comb_list = [(w, x) for w, x in zip(diff_omegas, self.labels)]
        k_ops2 = [np.sqrt(1-w) * self.model_gates[x] for w, x in comb_list]
        k_ops2.insert(0, np.sqrt(1 - sum(1-w for w, x in comb_list)) *
                            tensor(qeye(2), qeye(2)))
        k_ops1 = [np.sqrt(1-w) * self.model_gates[x] for w, x in comb_list[3:6]]
        k_ops1.insert(0, np.sqrt(1 - sum(1-w for w, x in comb_list[3:6])) *
                            tensor(qeye(2), qeye(2)))
        k_ops0 = [np.sqrt(1-w) * self.model_gates[x] for w, x in comb_list[0:3]]
        k_ops0.insert(0, np.sqrt(1 - sum(1-w for w, x in comb_list[0:3])) *
                            tensor(qeye(2), qeye(2)))
        return (k_ops0, k_ops1, k_ops2)
    
    def generate_super_ops(self, target_kops: List[Qobj] = None):
        ''' 
        This would work for any output from generate_kraus_ops.
        '''
        dim_list = [[2,2] for _ in range(self.num_qubits)]
        if target_kops is None:
            kops = self.kraus_ops_lam
        else:
            kops = target_kops
        super_op_tuple = (Qobj(kraus_to_super(kop), dims=[dim_list, dim_list])
                          for kop in kops)
        return tuple(super_op_tuple)
    
    def instantiate_target(self, target_phis):
        ''' Makes the target noise model. '''
        self.target = target_phis
        self.target_omegas = [0.5*(1 + np.exp(-2 * np.abs(x))) for x in self.target]
        self.kraus_ops_targ = self.generate_kraus_ops(self.target_omegas)
        self.super_ops_targ = self.generate_super_ops(self.kraus_ops_targ)
        
    def define_inverse(self):
        ''' Defines the map connecting the lambda noise model and the target noise model. '''
        if self.target is not None:
            self.inverse = [a - b for a, b in zip(self.lambdas, self.target)]
        else:
            raise ValueError('Need to instantiate the target noise model first.')
        self.inverse_omegas = mk_omega(self.inverse)
        overhead = [np.exp(sum([2*x if x > 0 else 0 for x in self.inverse[0:3]])),
                    np.exp(sum([2*x if x > 0 else 0 for x in self.inverse[3:6]])),
                    np.exp(sum([2*x if x > 0 else 0 for x in self.inverse]))]
        self.gammas = tuple(overhead)

    def make_plot(self, ttl: str = None, font_size: str = 20, save_file: str = None):
        '''
        Makes a bar plot of the model coefficients. Give ttl to set a title, and
        save_file to output the graph to a pdf named <save_file>.pdf.
        '''

        bw = 'black'

        fig, axes = plt.subplots(1,1, figsize=(9,6))

        axes.bar([n for n, _ in enumerate(self.lambdas)], self.lambdas, 
                 label='$\lambda_{k \in \mathcal{K}}$', tick_label=self.labels, color='0.25')
        axes.bar([n for n, _ in enumerate(self.target)], self.target, 
                 label='$\phi_{k \in \mathcal{K}}$', width=0.25, tick_label=self.labels, color='magenta')

        axes.legend(fontsize=font_size)

        if ttl is not None and type(ttl)==str:
            axes.set_title(ttl, fontsize=font_size, color=bw)
        axes.set_xlabel(r'$\mathcal{K}$', fontsize=font_size, color=bw)
        axes.set_ylabel(r'Values', fontsize=font_size, color=bw)
        axes.tick_params(colors=bw, which='both', labelsize=(font_size - 5))

        fig.tight_layout()
        plt.show()

        if save_file is not None and type(save_file)==str:
            with PdfPages(f'{save_file}.pdf') as export_pdf:
                export_pdf.savefig(fig)













