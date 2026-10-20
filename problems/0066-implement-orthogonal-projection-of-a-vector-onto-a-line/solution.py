import numpy as np 
def orthogonal_projection(v, L):
	v= np.asarray(v,dtype=float)
	L=np.asarray(L,dtype=float)
	denominator = np.dot(L,L)
	if denominator == 0 :
		return ValueError("denominator != 0 !!!")

	coeff = (np.dot(L,v) / denominator )
	projection = coeff*L
		
	return projection.tolist()
