"""
The entry to the Local Phase Image Quantum Encoding (LPIQE) Package for the image encoding-reconstruction problem.
It performs one experiment of image encoding on quantum back-end with method LPIQE and its reconstruction.
"""
from src.LPIQE_implementation import ILpqieImplementation
from src.LPIQE_representation import LpiqeRepresentation
from src.LPIQE_representation import OperatorType as ot
import numpy as np


class LpiqeReconstructionExecutor:
    """
    The class that represents the encoding-reconstruction for one image.
    """

    def __init__(self, executor, image_size: tuple, log_params, r_range=(0.25 * np.pi, 0.75 * np.pi)):
        """
        Constructor for the LPIQE Reconstruction experiment executor

        :param executor: the executor object, which is implementation on specific quantum back-end of the interface:
            LPIQE_implementation.ILpqieImplementation
        :param image_size: The size of a all images that will be processed with one instance of the executor
        :param log_params: The parameters for logging to a Quantum system / SDK / API
        :param r_range: The representation range. <br>
            Representation range is a connected subset of the local phase (phi) domain, which is [-pi, pi].
            Since the result is expressed by cos(phi) in the neighbourhood of 0 the values will be much
            more concentrated, which makes the final error greater. Therefore, it is reasonable to map x domain
            to such place in phi domain, where cos(phi) is most similar to linear function.<br>
            By default it is a range [0.25pi, 0.75pi]

        :raise TypeError: in case the given executor is not derived from LPIQE_implementation.ILpqieImplementation
        :raise Value error: in case the image size is not a pair.
        :raise ConnectionRefusedError: in case the loging to the quantum system was not possible
        """
        if not issubclass(type(executor), ILpqieImplementation):
            raise TypeError("Executor must be derived from LPIQE_implementation.ILpqieImplementation")
        if len(image_size) != 2:
            raise ValueError("Image size must be pair - two elements tuple.")
        self.__ex = executor
        if not self.__ex.login(log_params):
            raise ConnectionRefusedError('Login to the quantum system not possible')
        self.__repr = LpiqeRepresentation(image_size[0], image_size[1], r_range)
        self.__im_orig = np.ndarray(image_size)
        self.__im_recon = np.ndarray(image_size)
        self.__im_diff = np.ndarray(image_size)
        size = image_size[0] * image_size[1]
        self.__q_count = int(np.log2(image_size[0]) + np.log2(image_size[1])) + 1
        self.__im_operator = np.ndarray((size, size)).astype(complex)
        self.__q_circ = None
        self.__im_size = image_size

    def execute(self, image: np.ndarray, shots, print_info=True, print_recon=False):
        """
        This method executes encoding - reconstruction experiment for one image.

        :param print_recon: if True there is printed measurement eigen-states appearance histogram
        :param print_info: if True there is printed the information about consecutive steps of quantum computing
        :param image: Image to be encoded and reconstructed
        :param shots: The number of shots to be made on quantum back-end
        :return: nothing.
        """
        self.__repr.image_entry(image)
        self.__im_orig = self.__repr.original_image[0]
        self.__q_circ = self.__ex.homogeneous_superposition(self.__im_size)
        self.__q_circ = self.__ex.unitary_operator(self.__q_circ, self.__repr.operator_matrix[0])
        q_nr = 0
        if self.__repr.operator_type is ot.TP_UC_ANCILLA_RIGHT or self.__repr.operator_type is ot.TP_ANCILLA_RIGHT:
            q_nr = self.__q_count - 1
        self.__q_circ = self.__ex.set_hgate(self.__q_circ, q_nr)
        meas_nr = int(np.log2(self.__im_size[0])+np.log2(self.__im_size[0]))+1
        self.__q_circ = self.__ex.define_measurement(self.__q_circ, range(0, meas_nr), range(0, meas_nr))
        self.__repr.results = self.__ex.make_quantum_computation(self.__q_circ, shots, print_info)
        if not self.__repr.reconstruct(print_recon):
            raise RuntimeError('Image cannot be reconstructed')
        self.__im_recon = self.__repr.reconstructed_image[0]
        self.__im_diff = abs(self.__im_recon - self.__im_orig)

    @property
    def original_image(self):
        """
        Property for obtaining the original image. Should be called **after** execution of experiment

        :return: original image
        """
        return self.__im_orig

    @property
    def reconstructed_image(self):
        """
        Property for obtaining the reconstructed image. Should be called **after** execution of experiment

        :return: reconstructed image
        """
        return self.__im_recon

    @property
    def difference_image(self):
        """
        Property for obtaining the image of  difference between the original and reconstructed images
        from the last experiment.

        :return: absolute value difference image O-R, where O-original image, R - reconstructed image.
        """
        return self.__im_diff

    def std_dev(self):
        """
        Computes the standard deviation of image of differences between original and reconstructed images
        from last experiment.

        :return: standard deviation of difference image O-R, where O - original image, R - reconstructed image.
        """
        return np.std(self.__im_diff)

    def mean(self):
        """
        Computes the mean of image of differences between original and reconstructed images
        from last experiment.

        :return: mean of difference image O-R, where O - original image, R - reconstructed image.
        """
        return np.mean(self.__im_diff)

    def mse(self):
        """
        Computes the mean square error (MSE) of image of differences between original and reconstructed images
        from last experiment.

        :return: mse of difference image O-R, where O - original image, R - reconstructed image.
        """
        return np.mean(self.__im_diff ** 2)
