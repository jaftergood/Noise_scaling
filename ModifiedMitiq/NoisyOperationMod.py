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

import numpy as np
import numpy.typing as npt

import cirq
from cirq.value.linear_dict import _format_coefficient

from mitiq import QPROGRAM
from mitiq.interface import (
    convert_to_mitiq,
    CircuitConversionError,
    UnsupportedCircuitError,
)


class NoisyOperation:
    """An operation (or sequence of operations) which a noisy quantum computer
    can actually implement.p
    """

    def __init__(
        self,
        circuit: QPROGRAM,
        channel_matrix: Optional[npt.NDArray[np.complex64]] = None,
    ) -> None:
        """Initializes a NoisyOperation.

        Args:
            circuit: A short circuit which, when executed on a given noisy
                quantum computer, generates a noisy channel. It typically
                contains a single-gate or a short sequence of gates.
            channel_matrix: Superoperator representation of the noisy channel
                which is generated when executing the input ``circuit`` on the
                noisy quantum computer.

        Raises:
            TypeError: If ``ideal`` is not a ``QPROGRAM``.
        """
        self._native_circuit = deepcopy(circuit)

        # try:
        #     cirq_circuit, native_type = convert_to_mitiq(circuit)
        # except (CircuitConversionError, UnsupportedCircuitError):
        #     raise TypeError(
        #         "Failed to convert to an internal Mitiq representation"
        #         f"the input circuit:\n{type(circuit)}\n"
        #     )

        # self._circuit = cirq_circuit
        self._native_type = circuit
        # self.num_qubits = circuit.num_qubits

        dimension = 2**self._native_circuit.num_qubits

        if channel_matrix is None:
            self._channel_matrix = None

        elif channel_matrix.shape != (
            dimension**2,
            dimension**2,
        ):
            raise ValueError(
                f"Arg `channel_matrix` has shape {channel_matrix.shape}"
                " but the expected shape is"
                f" {dimension ** 2, dimension ** 2}."
            )
        self._channel_matrix = deepcopy(channel_matrix)

    @property
    def circuit(self) -> cirq.Circuit:
        """Returns the circuit of the NoisyOperation as a Cirq circuit."""
        return self._native_circuit

    @property
    def native_circuit(self) -> QPROGRAM:
        """Returns the circuit used to initialize the NoisyOperation."""
        return self._native_circuit

    # @property
    # def qubits(self) -> Tuple[cirq.Qid, ...]:
    #     return tuple(self._native_circuit.num_qubits())

    @property
    def num_qubits(self) -> int:
        return self._native_circuit.num_qubits

    @property
    def channel_matrix(self) -> npt.NDArray[np.complex64]:
        if self._channel_matrix is None:
            raise ValueError("The channel matrix is unknown.")
        return deepcopy(self._channel_matrix)

    def __add__(self, other: Any) -> "NoisyOperation":
        if not isinstance(other, NoisyOperation):
            raise ValueError(
                f"Arg `other` must be a NoisyOperation but was {type(other)}."
            )

        if self.num_qubits != other.num_qubits:
            raise NotImplementedError

        if self._channel_matrix is None or other._channel_matrix is None:
            matrix = None
        else:
            matrix = other._channel_matrix @ self._channel_matrix

        return NoisyOperation((self._native_circuit).compose(other._native_circuit), matrix)

    def __str__(self) -> str:
        return self._native_circuit.__str__()
