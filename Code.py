import numpy as np
import sklearn
from sklearn.metrics.pairwise import polynomial_kernel

# You are allowed to import any submodules of sklearn e.g. metrics.pairwise to construct kernel Gram matrices
# You are not allowed to use other libraries such as scipy, keras, tensorflow etc

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE submit.py

# DO NOT CHANGE THE NAME OF THE METHODS my_kernel, my_decode etc BELOW
# THESE WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THESE NAMES WILL CAUSE EVALUATION FAILURE

# You may define any new functions, variables, classes here
TUNED_DEGREE = 2
TUNED_COEF0 = 0.1


################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################
	K_poly = polynomial_kernel(Z1, Z2, degree=TUNED_DEGREE, coef0=TUNED_COEF0)
    
    # 2. Compute the x1 * x2 part.
    # X1 is (n1, 1), X2 is (n2, 1). X1 @ X2.T gives an (n1, n2) matrix.
	K_x = X1 @ X2.T
    
    # 3. Compute the final kernel: G = (x1 * x2) * K(Z1, Z2) + 1
    # This is an element-wise multiplication.
	G = K_x * K_poly + 1.0
	# Use this method to compute Gram matrices for your proposed kernel
	# Your kernel matrix will be used to train a kernel ridge regressor
	
	return G

def invert_arbiter_model(u):
    """
    Inverts a single 33-dim arbiter PUF model (u) to find
    32-dim non-negative delays (a, b, c, d).
    """
    # Initialize 32-element delay vectors
    a = np.zeros(32)
    b = np.zeros(32)
    c = np.zeros(32)
    d = np.zeros(32)

    # This implements the "hand-crafted solver" based on the method
    # derived in the Part 4 explanation.
    # We set beta_i = 0 for i=0..30.
    
    # This gives alpha_i = u_i for i=0..30
    # And we solve:
    #   a_i - b_i = u_i
    #   c_i - d_i = u_i
    # with non-negativity constraints.
    
    # Handle i = 0 to 30
    u_i = u[0:31]
    a[0:31] = np.maximum(u_i, 0)
    b[0:31] = np.maximum(-u_i, 0)
    c[0:31] = np.maximum(u_i, 0)
    d[0:31] = np.maximum(-u_i, 0)

    # Handle i = 31 (the last stage)
    # Here, alpha_31 = u_31 and beta_31 = u_32
    # We solve:
    #   a_31 - b_31 = u_31 + u_32
    #   c_31 - d_31 = u_31 - u_32
    
    a[31] = np.maximum(u[31] + u[32], 0)
    b[31] = np.maximum(-(u[31] + u[32]), 0)
    c[31] = np.maximum(u[31] - u[32], 0)
    d[31] = np.maximum(-(u[31] - u[32]), 0)
    
    return a, b, c, d

################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

    # Use this method to invert a PUF linear model to get back delays
    # w is a single 1089-dim vector (last dimension being the bias term)
    # The output should be eight 32-dimensional vectors
    
    # --- Step 1: "De-Kronecker" w -> u, v ---
    # We know w = u (tensor) v, where w is 1089 (33*33) and u, v are 33.
    # Reshaping w to W (33x33) gives W = u * v.T (a rank-1 matrix)
    W = w.reshape(33, 33)
    
    # We can use SVD to find the best rank-1 approximation.
    # W = U * S * Vh. The rank-1 approx is S[0] * U[:,0] * Vh[0,:]
    U, S, Vh = np.linalg.svd(W)
    
    # Split the singular value to get u and v
    s_root = np.sqrt(S[0])
    
    # u is the model for the first PUF (a,b,c,d)
    u = U[:, 0] * s_root
    
    # v is the model for the second PUF (p,q,r,s)
    v = Vh[0, :] * s_root
    
    # --- Step 2: Invert u and v to get delays ---
    # We use our helper function to solve the underdetermined linear system
    # for non-negative delays.
    a, b, c, d = invert_arbiter_model(u)
    p, q, r, s = invert_arbiter_model(v)
    
    return a, b, c, d, p, q, r, s

