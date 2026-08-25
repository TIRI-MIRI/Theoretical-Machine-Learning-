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

################################
# Non Editable Region Starting #
################################
def my_kernel( X1, Z1, X2, Z2 ):
################################
#  Non Editable Region Ending  #
################################
	
	return polynomial_kernel( X1, X2, degree = 4, coef0 = 0 ) + 1


################################
# Non Editable Region Starting #
################################
def my_decode( w ):
################################
#  Non Editable Region Ending  #
################################

	d = np.sqrt( w.shape[0] ).astype( int ) - 1 # The arbiter PUF model is 33 dimensional whereas the delay vectors should be 32 dimensional
	a = np.zeros( d, )		# delays must be non-negative but zeros are allowed
	b = np.zeros_like( a )
	c = np.zeros_like( a )
	d = np.zeros_like( a )
	p = np.zeros_like( a )
	q = np.zeros_like( a )
	r = np.zeros_like( a )
	s = np.zeros_like( a )
	
	return a, b, c, d, p, q, r, s
