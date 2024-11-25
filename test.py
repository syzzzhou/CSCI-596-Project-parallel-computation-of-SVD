import numpy as np
import math
from numpy.linalg import eig,eigvals
A=[[6,5,1,9,8,4],[8,5,2,4,6,9],[1,2,3,2,1,4]]
A=np.array(A)
ATA=np.dot(A,A.transpose())
eig,vec=eig(ATA)
print(eig)
print(vec)
