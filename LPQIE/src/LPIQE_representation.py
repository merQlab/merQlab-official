"""
LPIQE quantum image representation and reconstruction.

This module implements in Qiskit the method of storing and reconstruction of an image. The method name is Local Phase
Image Quantum Encoding (LPIQE). The method relies on creating the state vector [exp(i*p_0) ... exp(i*p_N)],
where i is the imaginary part of complex number, p_0, ..., p_N are intensities of pixel expressed in the range of [
0, 1] of float numbers. Atn: The image has to fulfill the condition width*height has to be the power of 2. The module
generates an evolution operator containing the image in four forms: * fit, where the state vector becomes a diagonal
of an operator. Hence, the operator has a shape width*height * ancilla left, where there is a tensor product of state
vector and ancilla (superposed) made with ancilla on the left side. It results with operator of shape 2*width*height
and the state vector is duplicated on the diagonal. * ancilla right, where there is a tensor product of state vector
and ancilla (superposed) made with ancilla on the right side  It results with operator of shape 2*width*height and
the diagonal has a form [exp(i*p_0), exp(i*p_0), ... exp(i*p_N), exp(i*p_N)] * ancilla left uncomputed,
where the procedure of uncomputation on the <<ancilla left>> form is made to have one copy of state vector on
diagonal. It has a form: [exp(i*p_0), ..., exp(i*p_N), 1, ..., 1] and shape 2*width*height. * ancilla right
uncomputed, where the procedure of uncomputation on the <<ancilla right>> form is made to have one copy of state
vector on diagonal. It has a form: [exp(i*p_0),1, ..., exp(i*p_N), 1] and shape 2*width*height. The <ancilla right>>
is the most proper form for reconstruction, while the others are for further computation purpose.

This module is not connected with implementation on specific quantum computer. It just prepars the LPQIE operator
for the given image and avails an image reconstruction basing on the results obtained from quantum back-end and
converted to a dictionary: (eigen-state, its appearance count).

The implementation on specific quantum computer should be made by interface ILpqieImplementation. Realization of it
is used in LpqieExecutor, which finally implements the concrete quantum circuit and run it on the given back-end.
"""

import warnings
from enum import Enum
from typing import Tuple

import numpy as np
from numpy import ndarray


class OperatorType(Enum):
    """Enumerator for LPIQE evolution operator.\n
    FIT:
        fit, where the state vector becomes a diagonal of an operator. Hence, the operator has a shape width*height
    TP_ANCILLA_LEFT:
        ancilla left, where there is a tensor product of state vector and ancilla (superposed) made with ancilla on the
        left side. It results with operator of shape 2*width*height and the state vector is duplicated on the diagonal.
    TP_ANCILLA_RIGHT:
        ancilla right, where there is a tensor product of state vector and ancilla (superposed) made with ancilla on the
        right side  It results with operator of shape 2*width*height and the diagonal has a form
        [exp(i*p_0), exp(i*p_0), ... exp(i*p_N), exp(i*p_N)]
    TP_UC_ANCILLA_LEFT:
        ancilla left uncomputed, where the procedure of uncomputation on the <<ancilla left>> form is made to have one
        copy of state vector on diagonal. It has a form: [exp(i*p_0), ..., exp(i*p_N), 1, ..., 1]
        and shape 2*width*height.
    TP_UC_ANCILLA_RIGHT:
        ancilla right uncomputed, where the procedure of uncomputation on the <<ancilla right>> form is made to have one
        copy of state vector on diagonal. It has a form: [exp(i*p_0),1, ..., exp(i*p_N), 1] and shape 2*width*height.
    """
    FIT = 1
    TP_ANCILLA_LEFT = 2
    TP_ANCILLA_RIGHT = 3
    TP_UC_ANCILLA_LEFT = 4
    TP_UC_ANCILLA_RIGHT = 5


class RepresentationState:
    RS_CLEAR = 0
    RS_IMAGE_ENTERED = 1
    RS_SYSTEM_SAMPLED = 2
    RS_IMAGE_RECONSTRUCTED = 3


class LpiqeRepresentation:
    """
    A class implementing LPIQE operator.
        Attributes:
            __width:int
                width of an image.
            __height:int
                height of an image. Attn: width*height has to be power of 2.
            __op:numpy.ndarray
                operator representation for current image.
            __originalImage: numpy.ndarray
                original, last entered, image
            __reconstructedImage: numpy.ndarray
                the reconstruction of last entered image.
        ---------

    """

    def __init__(self, w: int, h: int, r_range=(0.25 * np.pi, 0.75 * np.pi)):
        # operator's width
        if np.log2(w) != np.floor(np.log2(w)) or np.log2(h) != np.floor(np.log2(h)):
            raise ValueError('Image height and width must be powers of 2!')
        self.__width = w
        self.__height = h
        self.__widthQ = int(np.log2(w))
        self.__heightQ = int(np.log2(h))
        self.__op = np.zeros((2 * w * h, 2 * w * h)).astype(complex)
        self.__type = OperatorType.TP_UC_ANCILLA_RIGHT
        self.__originalImage = np.zeros((w, h)).astype(float)
        self.__reconstructedImage = np.zeros((w, h)).astype(float)
        self.__repr_state = RepresentationState.RS_CLEAR
        self.__results: dict[str, int] = dict[str, int]()
        self.__shots = 0
        self.__repr_range = r_range

    @property
    def representation_state(self):
        return self.__repr_state

    @representation_state.setter
    def representation_state(self, v):
        self.__repr_state=v

    @property
    def operator_type(self):
        return self.__type

    @operator_type.setter
    def operator_type(self, v:OperatorType):
        self.__type=v

    @property
    def image_size(self):
        """
        Getter for current image size.

        :return: tuple (width, height)
        """
        return self.__width, self.__height

    @image_size.setter
    def image_size(self, v):
        """Image size setter

        :param v: tuple (width, height)

        :raise IndexError: if input is not a pair.
        """
        if len(v) != 2:
            raise IndexError('Size of an image has to be pair. Now its length is ' + str(len(v)))
        self.__width = v[0]
        self.__height = v[1]

    @property
    def operator_matrix(self) -> (np.ndarray, bool):
        """
        Current LPQIE operator in matrix form

        :return: the tuple: [The complex matrix (numpy.ndarray) for of an operator for current image, b],
            b is True if image is entered and False in opposite case.
        """
        return self.__op, self.__repr_state >= RepresentationState.RS_IMAGE_ENTERED

    @property
    def representation_range(self):
        """
        Property for representation range. \n
        Representation range is a connected subset of the local phase (phi) domain, which is [-pi, pi].
        Since the result is expressed by cos(phi) in the neighbourhood of 0 the values will be much more concentrated,
        which makes the final error greater. Therefore, it is reasonable to map x domain to such place in phi domain,
        where cos(phi) is most similar to linear function.\n
        By default it is a range [0.25pi, 0.75pi]

        :return: Current representation range
        """
        return self.__repr_range

    @representation_range.setter
    def representation_range(self, v: tuple):
        """
        Setter for representation range.
        Representation range is a connected subset of the local phase (phi) domain, which is [-pi, pi].
        Since the result is expressed by cos(phi) in the neighbourhood of 0 the values will be much more concentrated,
        which makes the final error greater. Therefore, it is reasonable to map x domain to such place in phi domain,
        where cos(phi) is most similar to linear function. \n
        By default it is a range [0.25pi, 0.75pi]

        :param v: new representation range
        """
        self.__repr_range = v

    def __update_operator(self, x: float, i: int) -> int:
        """
        Updates operator on position "i" considering its current type.

        :param x: the value of current pixel

        :param i: the position in the operator

        :return: the proper next position in the operator.
        """
        self.__op[i, i] = np.exp(x * 1j)
        i = i + 1
        if self.__type == OperatorType.TP_ANCILLA_LEFT:
            jump = self.__width * self.__height
            self.__op[i + jump, i + jump] = np.exp(x * 1j)
        if self.__type == OperatorType.TP_UC_ANCILLA_LEFT:
            jump = self.__width * self.__height
            self.__op[i + jump, i + jump] = 1
        if self.__type == OperatorType.TP_ANCILLA_RIGHT:
            self.__op[i, i] = np.exp(x * 1j)
            i = i + 1
        if self.__type == OperatorType.TP_UC_ANCILLA_RIGHT:
            self.__op[i, i] = 1
            i = i + 1
        return i

    def image_entry(self, img: np.ndarray):
        """
        Entry an image and convert it to matrix form of an evolution operator LPIQE. The x is mapped to representation
        range domain of local phase.

        :param img: original image

        :return: nothing, just stores and image and operator,
            both available by properties unitary_operator and original_image

        :raise ValueError: if a pixel intensity is out of range [0,1]
        """
        self.__originalImage = img
        size = 2 * self.__width * self.__height

        self.__op = np.zeros((size, size)).astype(complex)
        i = 0

        for r in range(self.__width):
            for c in range(self.__height):
                x = img[r, c]
                if x < 0 or x > 1:
                    raise ValueError(
                        'Image for LPIQE representation should be one-channel image, where pixels are in [0,1] range. '
                        'We obtained: [' + str(r) + ', ' + str(c) + ']=' + str(x))
                else:
                    x = x * (self.__repr_range[1] - self.__repr_range[0]) + self.__repr_range[0]
                    i = self.__update_operator(x, i)
        self.__repr_state = RepresentationState.RS_IMAGE_ENTERED

    @property
    def original_image(self) -> (np.ndarray, bool):
        """
        Original image

        :return: The tuple: (original, last red, image in the form of numpy array, b)
            b is True if image is entered and False in opposite case.
        """
        return self.__originalImage, self.__repr_state >= RepresentationState.RS_IMAGE_ENTERED

    @property
    def results(self):
        return self.__results, self.__shots

    @results.setter
    def results(self, v: Tuple[dict[str, int], int]):
        """
        Registers the result of evolution made on quantum backend

        :param v: The Tuple, where:<br>
            - First element is the dictionary where keys are string representing measurement eigen-states and int
            represents the appearances count of given eigen-state <br>
            - Second element is the number of shots in the experiment which have generated the dictionary.

        :raise IndexError: if v is not a pair described above

        :raise AttributeError: if given dictionary is empty

        :return: True if succeed and False in opposite case
        """
        if len(v) != 2:
            raise IndexError('The results value must be pair (two-element tuple): '
                             '(the dictionary of eigen-states with its appearance count, number of shots)')
        if len(v[0]) == 0:
            raise AttributeError('The results are empty')
        self.__results = v[0]
        self.__shots = v[1]
        self.__repr_state = RepresentationState.RS_SYSTEM_SAMPLED

    def reconstruct(self, print_results: bool = False) -> bool:
        """
        Reconstruct an image after evolution and sampling.
        Reconstructed image can be obtained by reconstructedImage property.

        :param print_results: If True for each result following quantities are printed:<br>
         p0-probability<br>
         x - intermediate value<br>
         [u,v]=r the resulting value (r) of the pixel [u,v]

        :return: True if succeed and False in opposite case
        """
        if self.__repr_state != RepresentationState.RS_SYSTEM_SAMPLED:
            return False
        img = np.zeros((self.__width, self.__height)).astype(float)
        for result in sorted(self.__results.keys()):
            length = len(result)
            if result[length-1] == '0':
                p0 = self.__results[result] / self.__shots
                u = int(result[0:self.__widthQ], 2)
                v = int(result[self.__widthQ:self.__widthQ + self.__heightQ], 2)
                x = (2 ** (length+1) / 2) * p0 - 1  # 2**length - normalization factor
                if x < -1:
                    warnings.warn(
                        "The cos(p), p-experimentally determined probability for eigen state |" + result +
                        '> which maps pixel [' +
                        str(u) + ', ' + str(v) + '] is below -1.\n cos(p) changed to -1 to be able to compute arc cos!')
                    x = -1

                if x > 1:
                    warnings.warn(
                        "The cos(p), p-experimentally determined probability for eigen state |" + result +
                        '> which maps pixel [' +
                        str(u) + ', ' + str(v) + '] is over 1.\n cos(p) changed to 1 to be able to compute arc cos!')
                    x = 1

                img[u, v] = np.arccos(x)
                rl = self.__repr_range[1] - self.__repr_range[0]  # the representation range length
                img[u, v] = img[u, v] / rl - self.__repr_range[0] / rl  # inverse of mapping to representation range
                if print_results:
                    print('p(|' + result + '>)=' + str(p0), 'x=' + str(x),
                          '(' + str(u) + ',' + str(v) + ')=' + str(img[u, v]))
        self.__reconstructedImage = img
        return True

    @property
    def reconstructed_image(self) -> Tuple[ndarray, bool]:
        """
        Reconstructed image

        :return: The tuple: (image after the process of evolution, quantum sampling and reconstruction, b)<br>
            b is True if image is entered and False in opposite case.
        """
        return self.__reconstructedImage, self.__repr_state >= RepresentationState.RS_IMAGE_RECONSTRUCTED
