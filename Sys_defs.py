
import numpy as np
import qutip as qt
from math import factorial
from scipy.special import hermite
import sympy as sym
import scipy.integrate as integ

from importlib import reload
import Classes as Cls
reload(Cls)

# Function for squeezed coherent state
def sqcoh_magnon(alpha0,sqexp0,sqdir0,hil_size=40,no_samples=1e4,ampl='none',sq_exp='none'):

	Sys = Cls.Tomo()
	Sys.ampl = ampl
	Sys.sq_exp = sq_exp

	def ph_prob(a,phi):
		mean = 2*np.sin(ampl) * (alpha0*np.exp(-1j*phi)).real
		sigma_phi_sq = sqexp0**2 * np.sin(phi-sqdir0)**2 + np.cos(phi-sqdir0)**2/sqexp0**2
		var = np.cos(ampl)**2/sq_exp**2 + np.sin(ampl)**2*sigma_phi_sq**2

		quad_form = (a-mean)**2/(2*var)
		pre_fac = 1/(np.sqrt(2*np.pi*var))
		return pre_fac*np.exp(-quad_form)

	Sys.ph_prob = ph_prob

	amax = 3*np.abs(alpha0)**2  
	var_min = (np.cos(ampl)**2)/sq_exp**2 + np.sin(ampl)**2/sqexp0**2
	pmax = 1/(np.sqrt(2*np.pi*var_min))

	Sys.data = Sys.data_from_prob_ph(pmax = pmax, amax = amax, no_samples = no_samples)
		
	return Sys

# Function for cat state
def cat_magnon(alpha0,phi0,hil_size=40,no_samples=1e4,ampl='none',sq_exp='none'):

	Sys = Cls.Tomo()
	Sys.ampl = ampl
	Sys.sq_exp = sq_exp

	var = (np.cos(ampl)**2)/sq_exp**2 + np.sin(ampl)**2
	Pre_fac = 1/(1+np.exp(-2*np.abs(alpha0)**2)) * 1/np.sqrt(2*np.pi*var)

	def ph_prob(a,phi):
		z = alpha0*np.exp(-1j*phi)*np.sin(ampl)
		Rs = z.real
		Is = z.imag
		T1 = np.exp( - (a**2 + 4*Rs**2 )/(2*var) )
		T2_1 = np.cosh(2*a*Rs/var)
		T2_2 = np.exp(- (2 * np.abs(alpha0)**2 * np.cos(ampl)**2)/(sq_exp**2*var) ) * np.cos(2*a*Is/var - phi0)
		return Pre_fac * T1 * (T2_1 + T2_2)
	
	Sys.ph_prob = ph_prob

	amax = 3*np.abs(alpha0)**2
	pmax = 2*Pre_fac
	Sys.data = Sys.data_from_prob_ph(pmax = pmax, amax = amax, no_samples = no_samples)
	
	Sys.mag_state = (qt.coherent(hil_size,alpha0) + qt.coherent(hil_size,-alpha0)).unit()

	return Sys

## Function to call for creating a general state of magnons
def general_magnon(vec,hil_size=40,no_samples=1e4,ampl='none'):

	raise ValueError("Forbidden entry. Try not to squeeze in.")

	alpha = np.array(vec)
	max_m = len(alpha)
	
	if np.abs(np.linalg.norm(alpha) - 1) > 0.01:
		raise ValueError("The coefficients are not normalized")
	
	if len(alpha) > hil_size/2:
		raise ValueError("Increase the hilbert space dimensions")
	
	Sys = Cls.Tomo()
	Sys.ampl = ampl
	
	## Calculate the probability function for a given magnon state
	### Symbolic method: much faster but seems to gives some numerical inaccuracy.
	A,P,M,Th = sym.symbols('a phi m theta') # Defining symbols
	Alpha = [sym.Symbol('alpha_%s'%n) for n in range(max_m)] # A symbolic array to be replaced by alpha

	Arg = (sym.cos(Th)*M + sym.sin(Th)*A)/sym.sqrt(2)  # The typical argument appearing in the convolution

	I = lambda p,q: sym.integrate(sym.exp(-M**2/2) * sym.hermite(p,Arg) * sym.hermite(q,Arg),(M,-sym.oo,sym.oo) )  # The integrals to calculate
	
	prob_opt_sym = sym.exp(-A**2/2)/(2*sym.pi) * sum(
		[Alpha[p] * np.conj(Alpha[q]) * sym.exp(-sym.I*sym.Integer(p-q)*P)/sym.sqrt(2**(p+q) * factorial(p) * factorial(q)) * I(p,q) 
		for p,q in np.ndindex(max_m,max_m)]  # The symbolic expression
	)
	prob_opt_sym = sym.simplify(prob_opt_sym)
	
	to_python_funcs = sym.lambdify([A,P,Th] + Alpha,prob_opt_sym)  # Converts a symbolic expression into a python function with all symbols becoming arguments

	Sys.ph_prob = lambda a,phi: to_python_funcs(*([a,phi,np.pi/4] + list(vec))).real  # Keeping a,phi undefined while putting the value of Th=ampl and Alpha=vec

	### Numerical method: Excruciatingly slow
#	mag_prob_herm = lambda m,phi: sum([alpha[n]* np.exp(-1j*n*phi)*hermite(n)(m/np.sqrt(2))/np.sqrt(2**n * factorial(n)) for n in range(max_m)])
#	
#	Sys.ph_prob = lambda a,phi: np.exp(-a**2/2)/(2*np.pi) * integ.quad(
#		lambda m: np.exp(-m**2/2) * np.abs(
#			mag_prob_herm((np.cos(ampl)*m + np.sin(ampl)*a)/np.sqrt(2),phi)
#			)**2
#		,-np.inf,np.inf)[0]

	## These values should depend on 'alpha' but I don't have a formula for them though
	amax = 8
	pmax = 0.6

	Sys.data = Sys.data_from_prob_ph(pmax = pmax, amax = amax, no_samples = no_samples)

	alpha = np.pad(alpha,(0,hil_size - max_m))
	Sys.mag_state = qt.Qobj(alpha)

	return Sys


