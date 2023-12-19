import numpy as np

def Gradient(matrix1, matrix2):
    # Ensure the matrices are numpy arrays
    matrix1 = np.array(matrix1)
    matrix2 = np.array(matrix2)

    # Check that the matrices have the same shape
    if matrix1.shape != matrix2.shape:
        raise ValueError("The matrices must have the same shape")

    # Calculate the gradient
    gradient = matrix2 - matrix1

    return gradient
