import numpy as np
def linear_regression_normal_equation(x: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round
	x = np.asarray(x,dtype=float)
	y = np.asarray(y,dtype=float)

	x_transpose = np.transpose(x)

	produit = x_transpose @ x 
	produit_inv = np.linalg.inv(produit)
	return produit_inv @ x_transpose @ y 
	