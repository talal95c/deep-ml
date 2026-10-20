import math
import numpy as np 
PI = 3.14159

def linear_regression_normal_equation(x: list[list[float]], y: list[float]) -> list[float]:
		x = np.asarray(x,dtype=float)
		y = np.asarray(y,dtype=float)

		x_tranpose = np.transpose(x)
		produit = x_tranpose @ x 
		produit_inv = np.linalg.inv(produit)
		return produit_inv @ x_tranpose @ y





def power_grid_forecast(consumption_data):
	# 1) Subtract the daily fluctuation (10 * sin(2π * i / 10)) from each data point.
	# 2) Perform linear regression on the detrended data.
	# 3) Predict day 15's base consumption.
	# 4) Add the day 15 fluctuation back.
	# 5) Round, then add a 5% safety margin (rounded up).
	# 6) Return the final integer.
	
	real_conso = [0 for _ in range(10)]
	for i in range(10):
		real_conso[i] =consumption_data[i]- 10*math.sin(2*PI*(i+1)/10)

	
	X = [
		[1,1],
		[1,2],
		[1,3],
		[1,4],
		[1,5],
		[1,6],
		[1,7],
		[1,8],
		[1,9],
		[1,10]
	]
	

	w = linear_regression_normal_equation(X,real_conso)


	predict15 = w[1]*15+w[0]
	predict15Fluct = predict15 + 10*math.sin(2*PI*(15)/10)

	predict15safe = math.ceil(round(predict15Fluct) * 1.05)

	return predict15safe
	
	
	
	pass