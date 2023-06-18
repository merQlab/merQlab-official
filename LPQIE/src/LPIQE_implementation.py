"""
Interface for creating the concrete implementation using arbitrary chosen quantum library.\n

Requirements for library:
-------------------------
- Creating a quantum circuit\n
- Able to create homogeneous superposition
- Able to apply arbitrary unitary operator given in the matrix form
- Able to apply Hadamard transformation for arbitrary chosen qubit
- Able to transpile such a circuit and execute it on the quantum back-end (simulator or real quantum device)
- Able to obtain appearance histogram for chosen measurement basis.
"""
from typing import Any, Tuple

import numpy as np


class ILpqieImplementation:

    def login(self, parameters) -> bool:
        """
        Abstract method that is used for loging to a remote quantum system.

        :param parameters: the parameters needed for loging. The type is not specified, since in different systems
            can be different types and count of parameters

        :return: result of loging procedure
        """
        pass

    def homogeneous_superposition(self, image_size: tuple) -> Any:
        """
        Abstract method that creates the concrete circuit according to an implementation.
        It must provide the homogeneous superposition through all qubits (incl. ancillas).
        The qubit count must be equal to N=log2(width)+log2(height)+1.

        :param image_size: the size of an image that should be used for computing the qubit count (like above)

        :return: circuit implementing homogeneous superposition through N qubits.
        """
        pass

    def unitary_operator(self, circuit, matrix_form: np.ndarray) -> Any:
        """
        Converts the LPIQE operator, given in the matrix_form, to a form proper for concrete implementation in quantum
        system. Then sets the operator in a given circuit and returns modified circuit.

        :param circuit: <i>quantum circuit</i> made with a method <i>homogeneous_superposition()</i>
        :param np.ndarray matrix_form: Input matrix representing an operator.
            Use property: LpiqeRepresentation.operator_matrix
            for obtaining matrix form of an operator for the given image.

        :return: Unitary operator of the type defined in concrete quantum library.
        """
        pass

    def set_hgate(self, circuit, n: int) -> Any:
        """
        Sets the Hadamard gate for the n-th qubit in a given circuit and returns it after this operation

        :param circuit: <i>quantum circuit</i> made with a method <i>homogeneous_superposition()</i>
        :param n: the nr of a qubit, which hadamard gate act on.
        :return: <i>Circuit</i> that was given as an input after applying hadamard gate to the <i>qubit_nr</i> qubit
        """
        pass

    def define_measurement(self, circuit, q, c) -> Any:
        """
        Adds measurement gate to quantum circuit

        :param circuit: <i>quantum circuit</i> made with a method <i>homogeneous_superposition()</i>
        :param q: a sequence of qubits to be measured
        :param c: a sequence of bits for result
        :return: a circuit after adding the measurement gates
        """
        pass

    def make_quantum_computation(self, circuit, shots: int, print_info: bool = True) -> Tuple[dict[str, int], int]:
        """
        Executes the given quantum circuit on the quantum backend and returns the appearance histogram for measurement
        eigen-states.

        :param circuit: Circuit to be executed
        :param shots: The number of shots to be executed
        :param print_info: Prints the information about progress of computation
        :return: the tuple where on the first place is the dictionary representing the appearance histogram and on the
            second place is the shot count that was made in reality.
        """
        pass
