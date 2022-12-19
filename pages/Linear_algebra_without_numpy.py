from typing import List
from typing_extensions import assert_never

from numpy import multiply
Vector = List[float]

height_weight_age = [70, # inches,
170, # pounds,
40 ] # years
grades = [95, # exam1
80, # exam2
75 # exam3
]

def add(v: Vector, w:Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [vi+wi for vi, wi in zip(v, w) ]

def subtract(v:Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return [vi-wi for vi, wi in zip(v,w)]



def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Check that vectors is not empty
    assert vectors, "no vectors provided!"
    # Check the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"
    # the i-th element of the result is the sum of every vector[i]
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

#print(vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]))

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Multiplies every element by c"""
    return [c * v_i for v_i in v]

#print(scalar_multiply(2, [2, 22]))

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    assert vectors, "no list of vectors provided!"
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

#x = [[1, 2], [3, 4], [5, 6], [7, 8]]
#print(vector_mean(x))

def dot(v: Vector, w: Vector) -> float:
    """Computes v_1 * w_1 + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be same length"
    return sum(v_i * w_i for v_i, w_i in zip(v, w))

#print(dot([1, 2, 3], [4, 5, 6]))


def sum_of_squares(v: Vector) -> float:
    assert v, "not vector provided"
    return dot(v,v)

#print(sum_of_squares([1, 2, 3]))

from math import acos, sqrt, pi
def magnitude(v: Vector) -> float: #to compute its magnitude (or length)
    return sqrt(sum_of_squares(v))

#print(magnitude([3,4]))

def squared_distance(v : Vector, w:Vector) -> float:
    return magnitude(subtract(v,w))

#print(squared_distance([1, 2, 3], [4, 5, 6]))

#################################################################################################################
#################################################               #################################################
#################################################   Les Matrices ################################################
#################################################               #################################################
#################################################################################################################

# Another type alias
Matrix = List[List[float]]


B = [               # B has 3 rows and 2 columns
        [1, 2], 
        [3, 4],
        [5, 6]
    ]
#nRows = len(B)
#nCols = len(B[0])
#print(f"Matrix B shape : ({nRows}, {nCols})")


from typing import Tuple
def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of columns of A)"""
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0 # number of elements in first row
    return num_rows, num_cols

#print(shape(B))


def get_row(A : Matrix, i : int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    assert i in range(len(A)), f"nRows Max = {shape(A)[0]}"
    return A[i]

#print(get_row(B,5))

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the j-th column of A (as a Vector)"""
    assert j in  range(len(A[0])), f"nCols Max = {shape(A)[1]}"
    return [A_i[j] # jth element of row A_i
        for A_i in A] # for each row A_i

#print(get_column(B,2))



from typing import Callable
def make_matrix(num_rows: int,
    num_cols: int,
    entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return [[entry_fn(i, j) # given i, create a list
    for j in range(num_cols)] # [entry_fn(i, 0), ... ]
    for i in range(num_rows)] # create one list for each i


def identity_matrix(n: int) -> Matrix:
    """Returns the n x n identity matrix"""
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

#x = identity_matrix
#for i in range(len(x(5))): print(f"row[{i}] : {x(5)[i]}")





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

def L1_norm(v: Vector) -> float:
    """ Returns the L1 vector Norm (sometimes called the taxicab norm or the Manhattan norm) """
    assert v, "not vector provided"
    return sum(abs(v_i) for v_i in v)

#x = [-5,-6]
#print(L1_norm(x))



################## 
################## 
################## Vector Max Norm (sometimes called maximum norm)
################## maxnorm(v) = ||v||infinity =  max(|a1|, |a2|, |a3|, ...)
################## 

def maxnorm(v:Vector) -> float:
    """Returns the max norm of a vector v"""
    assert v, "not vector provided"
    return max(abs(v_i) for v_i in v)

#x = [1,2,3]
#print(maxnorm(x))


################## 
################## 
################## L2 Norm (also known as the Euclidean norm / the Euclidean distance from the origin)
################## l2(v) = ||v||2 = sqrt(a1^2 + a2^2 + a3^2+ ...)
################## 



L2_norme = magnitude([1,2,3])
#print(L2_norme)



################## 
################## 
################## Lp Norm 
################## Lp(v) = ||v|Lp = (a1^p + a2^p + a3^p+ ...)^(1/p)
################## 

def Lp_norme(v:Vector, p:int) -> float:
    """Returns the max norm of a vector v"""
    assert v, "not vector provided"
    assert p , "p must be rovided as integer"
    assert p>0, "p must be grater than 0"
    sum_v_p = sum([abs(v_i)**p for v_i in v])
    return sum_v_p**(1/p)


#print(Lp_norme([1,2,3],3))



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

def angle_between(v:Vector, w:Vector) -> float:
    """ Returns the angle in radians between vectors 'v' and 'w'::

            >>> angle_between([1, 0, 0], [0, 1, 0])
            Angle in radians = 1.5707963267948966 
            Angle in degrees = 90.0
    """
    assert v and w, "v and w must be vectors"
    ang_vw =  acos(dot(v,w)/(magnitude(v)*magnitude(w)))
    return f"Angle in radians = {ang_vw} \n Angle in degrees = {ang_vw*(360/(2*pi))}"    #from math import pi

w = [5,12]
v = [-6,8]
#print(angle_between(v,w))


################## 
################## 
################## Unit vector from vector
################## Unit vectors are vectors whose magnitude is exactly 1 unit
################## Unit vector of vector u is a vector in the same direction of u but with magnitude equals to 1
################## To find a unit vector with the same direction as a given vector, we divide the vector by its magnitude
################## The normalized vector û of a non-zero vector u is the unit vector in the direction of u
################## û = u/L2(u)

def unit_vector(u : Vector) -> Vector:
    """ Returns the unit vector from vector u::
            >>> unit_vector([-8,5,10])
            [-0.5819143739626463, 0.363696483726654, 0.727392967453308]
    """
    return  scalar_multiply(1/magnitude(u), u)

print(unit_vector([-8,5,10]))