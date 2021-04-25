import cv2
import numpy as np 
from scipy.fftpack import dct, idct
def reflectSuppress(Im, h, epsilon):
	Y = Im.astype(np.float64)/255.
	m,n, r = Im.shape
	T = np.zeros_like(Y)
	Y_Lapplacian_2 = np.zeros_like(Y)
	for dim in range(r):
		GRAD = grad(Y[:, :, dim])
		GRAD_x = GRAD[:, :, 0]
		GRAD_y = GRAD[:, :, 1]
		GRAD_norm = np.sqrt(GRAD_x**2 + GRAD_y**2)
		GRAD_norm_thresh = GRAD_norm.copy()
		
		GRAD_norm_thresh[np.abs(GRAD_norm_thresh <= h)] = 0
		ind = (GRAD_norm_thresh == 0)
		
		GRAD_x[ind] = 0
		GRAD_y[ind] = 0
		GRAD_thresh = np.zeros((m, n, 2))
		GRAD_thresh[:, :, 0] = GRAD_x
		GRAD_thresh[:, :, 1] = GRAD_y
		
		Y_Lapplacian_2[:, :, dim] = div(grad(div(GRAD_thresh)))
	rhs = Y_Lapplacian_2 + epsilon*Y

	for dim in range(r):
		T[:, :, dim] = PoissonDCT_variant(rhs[:, :, dim], 1, 0, epsilon)
		
	return T

def PoissonDCT_variant(rhs, mu, lamda, epsilon):
	M, N = rhs.shape[:2]
	k = np.arange(1, M+1).astype(np.float64)
	l = np.arange(1, N+1).astype(np.float64)
	k = k.reshape(M, 1)
	
	eN = np.ones((1, N)).astype(np.float64)
	eM = np.ones((M, 1)).astype(np.float64)
	
	k = np.cos(np.pi/M * (k-1))
	l = np.cos(np.pi/N * (l-1))
	
	k = np.kron(k, eN)
	l = np.kron(eM, l)
	kappa = 2*(k+l-2)
	const = mu * kappa**2 - lamda * kappa + epsilon
	#u = dct(rhs, norm='ortho')
	u = dct(dct(rhs, axis=0, norm= "ortho"), axis=1, norm="ortho")
	u = u / const
	#u = idct(u)
	u = idct(idct(u, axis=0, norm= "ortho"), axis=1, norm="ortho")
	return u
	
	


# compute the gradient of a 2D image array
def grad(A):
	m,n = A.shape[:2]
	B = np.zeros((m, n, 2))
	
	Ar = np.zeros((m, n))
	Ar[:, :n-1] = A[:, 1:n]
	Ar[:, n-1] = A[:, n-1]

	Au = np.zeros((m, n))
	Au[:m-1, :] = A[1:m, :]
	Au[m-1, :] = A[m-1, :]

	B[:, :, 0] = Ar - A
	B[:, :, 1] = Au - A

	return B

def div(A):

	m, n = A.shape[:2]
	B = np.zeros((m, n))
	
	T = A[:, : , 0]
	T1 = np.zeros((m, n))
	T1[:, 1:n] = T[:, 0:n-1]
	
	B = B+T - T1
	
	T = A[:, :, 1]
	T1 = np.zeros((m, n))
	T1[1:m, :] = T[:m-1, :]
	
	B = B+T-T1
	return B

if __name__ == "__main__":
	image = cv2.imread("./example/toy_example.jpg")
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	h =  0.11
	epsilon = 1e-8
	new = reflectSuppress(image, h, epsilon)
	new = new*255
	new = new.astype(np.uint8)
	new = cv2.cvtColor(new, cv2.COLOR_RGB2BGR)
	cv2.imwrite("res2.jpg", new)
	cv2.waitKey(0)	