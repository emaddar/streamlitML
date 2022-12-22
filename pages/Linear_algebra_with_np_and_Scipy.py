#https://primer-computational-mathematics.github.io/book/c_mathematics/linear_algebra/5_Linear_Algebra_in_Python.html

import numpy as np
import numpy.linalg as la

#print('identity(5) = \n', np.identity(5))  # square 5x5 matrix
#print('\neye(4, 5) = \n', np.eye(4, 5))  # 4x5 matrix
#print('\neye(4, 5, -1) = \n', np.eye(4, 5, -1))  # 1st lower diagonal
#print('\neye(4, 5, 2) = \n', np.eye(4, 5, 2))  # 2nd upper diagonal


M = np.array([[1, 2],
              [3, 4]])
v = np.array([6, 8])

#print('diag(M) = ', np.diag(M))
#print('diag(v) = \n', np.diag(v))


###################################################################################################
#######################################  Triangular matrix  #######################################
###################################################################################################

M = np.arange(1, 17).reshape(4, 4)
#print('M = \n', M)

#print('\ntriu(M) = \n', np.triu(M))
#print('\ntriu(M, -1) = \n', np.triu(M, -1))
#print('\ntril(M, 1) = \n', np.tril(M, 1))


###################################################################################################
#######################################    Sparse matrices  #######################################
###################################################################################################

# ................A sparse matrix is a matrix with mostly zero-valued entries.
#.................SciPy allows us to build such matrices and do operations on them with the scipy.sparse package. 


from scipy.sparse import coo_matrix


a, b, c = [1] * 9, [2] * 10, [3] * 9

A = np.diag(a, -1) + np.diag(b, 0) + np.diag(c, 1)
#print(A)

spA = coo_matrix(A)
#print(spA)




###############################################################################
###############################################################################
##########################     Vector    Norm    ##############################
###############################################################################
###############################################################################

################## 
################## 
################## Vector L1 Norm (sometimes called the taxicab norm or the Manhattan norm)
################## L1 (Vector) = ||v||1 = Somme(|a_i|) = |a1| + |a2| + |a3| + ...
################## 


from numpy.linalg import norm
a = np.array([-5,-6])
#print(a)
l1 = norm(a, 1)
#print(l1)


################## 
################## 
################## Vector Max Norm (sometimes called maximum norm)
################## maxnorm(v) = ||v||infinity =  max(|a1|, |a2|, |a3|, ...)
################## 

from numpy.linalg import norm
a = np.array([1, 2, 3])
#print(a)
maxnorm = norm(a, np.inf)
#print(maxnorm)



################## 
################## 
################## L2 Norm (also known as the Euclidean norm / the Euclidean distance from the origin)
################## l2(v) = ||v||2 = sqrt(a1^2 + a2^2 + a3^2+ ...)
################## 


a = np.array([1, 2, 3])
#print(a)
L2 = norm(a)
#print(L2)


################## 
################## 
################## Lp Norm 
################## Lp(v) = ||v|Lp = (a1^p + a2^p + a3^p+ ...)^(1/p)
################## 


a = np.array([1, 2, 3])
#print(a)
L3 = norm(a, ord = 3)
#print(L3)



###############################################################################
###############################################################################
##########################     Two Vectors       ##############################
###############################################################################
###############################################################################

################## 
################## 
################## Angle between two vectors (cosine rule) / theta \
################## cos(theta) =dot(v,w)/(L2(v)*L2(w))
##################  theta = arccos( dot(v,w)/(L2(v)*L2(w)) )

import math
def angle_between(v, w):
    """ Returns the angle in radians between vectors 'v' and 'w'::

            >>> angle_between(np.array([1, 0, 0])  , np.array([0, 1, 0]) )
            Angle in radians = 1.5707963267948966 
            Angle in degrees = 90.0
    """
    ang_vw = math.acos((v @ w)/(norm(v)*norm(w)))
    return f"Angle in radians = {ang_vw} \n Angle in degrees = {ang_vw*(360/(2*np.pi))}"

w = np.array([1, 0, 0])
v = np.array([0, 1, 0])
#print(angle_between(v,w))


################## 
################## 
################## Unit vector from vector
################## Unit vectors are vectors whose magnitude is exactly 1 unit
################## Unit vector of vector u is a vector in the same direction of u but with magnitude equals to 1
################## To find a unit vector with the same direction as a given vector, we divide the vector by its magnitude
################## The normalized vector û of a non-zero vector u is the unit vector in the direction of u
################## û = u/L2(u)

def unit_vector(u):
    """ Returns the unit vector from vector u::
            >>> unit_vector( np.array([-8,5,10]) )
            [-0.5819143739626463, 0.363696483726654, 0.727392967453308]
    """
    return  u*(1/norm(u))

u = np.array([-8,5,10])
print(unit_vector(u))