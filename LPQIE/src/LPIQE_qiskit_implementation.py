from typing import Any, Tuple
import numpy as np
from qiskit import IBMQ, QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
from qiskit.quantum_info import Operator

from src.LPIQE_implementation import ILpqieImplementation


class LpiqeQiskit(ILpqieImplementation):
    def __init__(self):
        self.__N = None
        self.__backend = None

    def login(self, parameters) -> bool:
        """
        Abstract method that is used for loging to a remote quantum system.

        :param parameters: the parameters needed for loging. Parameters should contain:<br>
            - [0]: bool, True in case of saving account, False in opposite case <br>
            - [1]: account TOKEN, that can be obtained from IBM Q web system. <br>
            - [2]: provider info: hub <br>
            - [3]: provider info: group <br>
            - [4]: provider info: project <br>
            - [5]: backend

        :return: result of loging procedure
        """
        # IBMQ.save_account(
        #     '6a780c166609fa9e4de81bb085725b615135a37caf29e4fff82e9d3e3ba864662469ab94fcaa8165e03318df4aa3887b98799b46b7d5979941524a767a9c0f3b',
        #     overwrite=True)
        if parameters[0]:
            IBMQ.save_account(parameters[1], overwrite=True)
        IBMQ.load_account()  # Load account from disk
        # provider = IBMQ.get_provider(hub='ibm-q-psnc', group='internal', project='default')
        provider = IBMQ.get_provider(hub=parameters[2], group=parameters[3], project=parameters[4])
        # self.__backend = provider.get_backend('simulator_statevector')
        self.__backend = provider.get_backend(parameters[5])
        return True

    def homogeneous_superposition(self, image_size: tuple) -> Any:
        """
        Abstract method that creates the concrete circuit according to an implementation.
        It must provide the homogeneous superposition through all qubits (incl. ancillas).
        The qubit count must be equal to N=log2(width)+log2(height)+1.

        :param image_size: the size of an image that should be used for computing the qubit count (like above)

        :return: circuit implementing homogeneous superposition through N qubits.
        """
        self.__N = int(np.log2(image_size[0]) + np.log2(image_size[1])) + 1
        c = QuantumCircuit(QuantumRegister(self.__N, name="p"),
                           ClassicalRegister(self.__N, name="c"))

        c.h(range(0, self.__N - 1))
        return c

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
        op = Operator(matrix_form)
        circuit.unitary(op, range(0, self.__N), label='img')
        return circuit

    def set_hgate(self, circuit, n: int) -> Any:
        """
        Sets the Hadamard gate for the n-th qubit in a given circuit and returns it after this operation

        :param circuit: <i>quantum circuit</i> made with a method <i>homogeneous_superposition()</i>
        :param n: the nr of a qubit, which hadamard gate act on.
        :return: <i>Circuit</i> that was given as an input after applying hadamard gate to the <i>qubit_nr</i> qubit
        """
        circuit.h(n)
        return circuit

    def define_measurement(self, circuit, q, c) -> Any:
        """
        Adds measurement gate to quantum circuit

        :param circuit: <i>quantum circuit</i> made with a method <i>homogeneous_superposition()</i>
        :param q: a sequence of qubits to be measured
        :param c: a sequence of bits for result
        :return: a circuit after adding the measurement gates
        """
        if len(q) is not len(c):
            raise IndexError('The set of measured qubits and classical bits for results must have the same size')
        if len(q) > self.__N:
            raise ValueError('The size of qubit\'s set to be measured cannot be greater then '
                             'qubit count existing in circuit')
        circuit.measure(q, c)
        return circuit

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
        if print_info:
            print('Quantum computation started: ')
            print('1. Transpilation started...')
        tc = transpile(circuit, self.__backend)
        if print_info:
            print('   Circuit is transpiled.\n2. Preparing a job.')

        job = execute(tc, self.__backend, shots=shots)
        if print_info:
            print('   Job is prepared.\n3. Sending job to execution')
        result = job.result()

        counts = result.get_counts(circuit)
        # print(counts)
        if print_info:
            print('  Job is executed')
        return [counts, shots]
