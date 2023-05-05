# Copyright (C) 2020 Unitary Fund (original code) -- This is a rework
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Types used in probabilistic error cancellation."""

from copy import deepcopy
from typing import Any, List, Optional, Tuple
import warnings

import qiskit
from qiskit import QuantumCircuit

import numpy as np
import numpy.typing as npt

import cirq
from cirq.value.linear_dict import _format_coefficient

from mitiq import QPROGRAM

from ModifiedMitiq.NoisyOperationMod import NoisyOperation

class OperationRepresentation:
    """A decomposition (basis expansion) of an operation or sequence of
    operations in a basis of noisy, implementable operations.
    """

    def __init__(
        self,
        ideal: QPROGRAM,
        noisy_operations: List[NoisyOperation],
        coeffs: List[float],
        is_qubit_dependent: bool = True,
    ) -> None:
        """Initializes an OperationRepresentation.

        Args:
            ideal: The ideal operation desired to be implemented.
            basis_expansion: Representation of the ideal operation in a basis
                of `NoisyOperation`s.
            is_qubit_dependent: If True, the representation
                corresponds to the operation on the specific qubits defined in
                `ideal`. If False, the representation is valid for the same
                gate even if acting on different qubits from those specified in
                `ideal`.

        Raises:
            TypeError: If all keys of `basis_expansion` are not instances of
                `NoisyOperation`s.
        """
        if not all(isinstance(o, NoisyOperation) for o in noisy_operations):
            raise TypeError(
                "All elements of `noisy_operations` must be "
                "of type `ModifiedMitiq.NoisyOperationMod.NoisyOperation`."
            )

        if not all(isinstance(c, float) for c in coeffs):
            raise TypeError("All elements of `coeffs` must be floats.")

        self._native_ideal = deepcopy(ideal)
        # self._ideal, self._native_type = convert_to_mitiq(ideal)
        self._ideal, self._native_type = deepcopy(ideal), type(ideal)
        self._noisy_operations = noisy_operations
        self._coeffs = coeffs
        self._norm = sum(abs(c) for c in coeffs)
        self._distribution = [abs(c) / self._norm for c in coeffs]
        self.is_qubit_dependent = is_qubit_dependent
        self._validate()

    def _validate(self) -> None:
        """Validates initialization arguments."""
        if len(self._noisy_operations) != len(self._coeffs):
            raise ValueError(
                "`noisy_operations` and `coeffs` must have equal length"
                f" but {len(self._noisy_operations)}!={len(self._coeffs)}."
            )
        if not np.isclose(sum(self._coeffs), 1.0, atol=10**-4):
            warnings.warn("The sum of the coefficients is different from 1.")
        # for op in self._noisy_operations:
        #     if self._ideal.all_qubits() != op.circuit.all_qubits():
        #         raise ValueError(
        #             "The operation to represent acts on"
        #             f" {self._ideal.all_qubits()}. Noisy operations"
        #             f" must act on the same qubits but {op} acts on:"
        #             f" {op.circuit.all_qubits()}"
        #         )

    @property
    def ideal(self) -> qiskit.QuantumCircuit:
        return self._ideal

    @property
    def basis_expansion(self) -> List[Tuple[float, NoisyOperation]]:
        return [(c, o) for c, o in zip(self._coeffs, self._noisy_operations)]

    @property
    def noisy_operations(self) -> List[NoisyOperation]:
        return self._noisy_operations

    @property
    def coeffs(self) -> List[float]:
        """Returns the coefficients of the quasi-probability distribution."""
        return self._coeffs

    @property
    def norm(self) -> float:
        """Returns the 1-norm of the quasi-probability distribution."""
        return self._norm

    @property
    def distribution(self) -> List[float]:
        """Returns the probability distribution obtained from taking
        the absolute value and normalizing the quasi-probability distribution.
        """
        return self._distribution

    def sample(
        self, random_state: Optional[np.random.RandomState] = None
    ) -> Tuple[NoisyOperation, int, float]:
        """Returns a randomly sampled NoisyOperation from the basis expansion.

        Args:
            random_state: Defines the seed for sampling if provided.
        """
        if not random_state:
            rng = np.random
        elif isinstance(random_state, np.random.RandomState):
            rng = random_state  # type: ignore
        else:
            raise TypeError(
                "Arg `random_state` should be of type `np.random.RandomState` "
                f"but was {type(random_state)}."
            )

        idx = rng.choice(len(self.coeffs), p=self.distribution)
        coeff, noisy_op = self.basis_expansion[idx]
        return noisy_op, int(np.sign(coeff)), coeff

    def __str__(self) -> str:
        lhs = str(self._ideal) + " = "
        rhs = ""
        for c, circ in zip(self.coeffs, self.noisy_operations):
            c_str = _format_coefficient(".3f", c)
            if c_str:
                if c_str[0] not in ["+", "-"]:
                    c_str = "+" + c_str
                if self._ideal.num_qubits == 1:
                    # Print single-qubit circuits horizontally
                    rhs += f"{c_str}*({circ!s})"
                else:
                    # Print multi-qubit circuits vertically
                    rhs += "\n\n" + f"{c_str}\n{circ!s}"
        # Handle special cases as in cirq.value.linear_dict._format_terms()
        if not rhs:
            rhs = f"{0:.3f}"
        # Remove "+" in the first term of a single-qubit representation
        if rhs[0] == "+":
            rhs = rhs[1:]
        # Remove "+" in the first term of a multi-qubit representation
        if rhs[0:3] == "\n\n+":
            rhs = "\n\n" + rhs[3:]
        return lhs + rhs

    # def __eq__(self, other: Any) -> bool:
    #     """Checks if two representations are equivalent. This function returns
    #     True if the representations have the same ideal operation, the same
    #     coefficients and equivalent NoisyOperation(s) (same gates but not
    #     necessarily same channel_matrix since channel_matrix is optional).
    #     """
    #     if self.is_qubit_dependent != other.is_qubit_dependent:
    #         return False

    #     if self._native_type != other._native_type:
    #         return False

    #     if not self._ideal == other._ideal:
    #         return False

    #     if len(self.basis_expansion) != len(other.basis_expansion):
    #         return False

    #     for c_a, op_a in self.basis_expansion:
    #         found = False
    #         for c_b, op_b in other.basis_expansion:
    #             if op_a._circuit == op_b._circuit:
    #                 found = True
    #                 break
    #         if not found:
    #             return False
    #         if not np.isclose(c_a, c_b):
    #             return False
    #     return True