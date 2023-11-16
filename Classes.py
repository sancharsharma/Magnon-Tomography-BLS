# This file contains several untested functions which can be removed to make it cleaner.

import scipy.integrate as integ
from scipy.special import erf
import numpy as np
from math import factorial
import multiprocess as mp
import qutip as qt
import tqdm


class Tomo:
	
	def __init__(self):

		self.data = 'none'  # Optical data
		self.ph_state = 'none'  # Photon State
		self.mag_state = 'none'  # Magnon State
		self.ph_wig = 'none'  # Photon Wigner function
		self.mag_wig = 'none'  # Magnon Wigner function
		self.ph_prob = 'none'  # Photon probability density
		self.mag_prob = 'none'   # Magnon probability density
		self.ampl = 'none'  # Magnon Signal Amplitude, 'theta'
		self.sq_exp = 1  # Squeezing in the input, 'e^r'

	##################### Functions for converting among different representations of photons

	# Probability from Wigner function
	def prob_from_wig_ph(self,a,phi): 		
		return 'This is not implemented'
		if (self.ph_wig == 'none'):
			raise ValueError('Supply photon wigner function using self.ph_wig = lambda a: ...')
		
		def integrand(b):
			arg_R = (a*np.cos(phi) - b*np.sin(phi))/2
			arg_I = (b*np.cos(phi) + a*np.sin(phi))/2
			return self.wig(arg_R + 1j*arg_I)
	
		int_res = integ.quad(integrand,-np.inf,np.inf)[0]
		return int_res/(4*np.pi)
	
	# Optical data from photon probability density using the so-called rejection sampling. pmax: The maximum value of probability function, amax: The maximum value of observations, no_samples: number of samples to be generated
	def data_from_prob_ph(self,pmax = 1, amax = 6, no_samples = 5e3):  

		if (self.ph_prob == 'none'):
			raise ValueError('Supply photon probability function using self.ph_prob = lambda a,phi: ...')
		
		return _rejection_sampling_(self.ph_prob, pmax = pmax, amax = amax, no_samples = no_samples)
		
	# Wigner function from data using direct inversion. 'alpha' is the Wigner argument and 'kc' is the high wave-vector cut-off for the inversion to be well-defined
	def wig_from_data_ph(self,alpha,kc=5): 
		
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')

		def _Kernel_(x):
			if np.abs(x) < 0.01/kc:
				return kc**2
			else:
				return 2*(np.cos(kc*x) + kc*x*np.sin(kc*x) - 1)/x**2
	
		K_data = []

		for entry in self.data:
			a = entry[0]
			phi = entry[1]
			y = 2*(alpha.real*np.cos(phi) + alpha.imag*np.sin(phi)) - a
			K_data.append(_Kernel_(y))

		return np.mean(K_data)
	
	# Photon density Matrix from optical data using MLE. hil_size: Hilbert space of the output, rho_tol: error tolerance in the recursion, max_iter: maximum number of iterations
	def den_from_data_ph(self, hil_size=40, rho_tol=0.005, max_iter=100):
		
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')
		
		data = self.data

		print("Computing Projections")
		psi_data = [_psi_(entry,hil_size = hil_size) for entry in tqdm.tqdm(data)]
		Projs = [psi*psi.dag() for psi in psi_data]
		

		print("Fixed point iteration begins")
		return _fixed_pt_(Projs,rho_tol=rho_tol,max_iter=max_iter)


	##################### Functions when a magnon state is given

	# Optical data from magnon probability density using numerical convolution and the so-called rejection sampling. pmax: The maximum value of probability function, amax: The maximum value of observations, no_samples: number of samples to be generated
	def data_from_prob_mag(self, pmax = 1, amax = 6, no_samples = 5e3): # pmax: The maximum value of probability function, amax: The maximum value of observations, no_samples: number of samples to be generated
		return 'This is not tested'

		if (self.ampl == 'none'):
			raise ValueError('Supply amplification factor using self.ampl = ...')

		if (self.mag_prob == 'none'):
			if (self.ph_prob == 'none'):
				raise ValueError('Supply magnon probability function using self.mag_prob = lambda a,phi: ... or directly the output photon probability function self.ph_prob = lambda a,phi: ...')
			_prob_opt_ = self.ph_prob
		else:
			p_vac = lambda a,phi: np.exp(-a**2/2)/np.sqrt(2*np.pi) # Can introduce squeezing
			p_mag = self.mag_prob
			ampl = self.ampl
			def _prob_opt_(a,phi):
				return integ.quad(lambda m: p_vac(np.cos(ampl)*a - np.sin(ampl)*m,phi) * p_mag(np.cos(ampl)*m+np.sin(ampl)*a,phi),-np.inf,np.inf)[0]
	
		return _rejection_sampling_(_prob_opt_,pmax = pmax, amax = amax, no_samples = no_samples)


	# Wigner function of magnons from optical data using direct inversion. 'alpha' is the Wigner argument and 'b0' is a cut-off to remove "high frequency" noise.
	def wig_from_data_mag(self,alpha,b0=2):
		
		return "This generally gives terrible results. Use it only if you understand what this function does."

		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')

		if (self.ampl == 'none'):
			raise ValueError('Supply ampl using self.ampl = ...]')

		ampl = self.ampl

		def _Kmod_exp_(y): 
			T1 = 2*(np.exp(b0**2/2)*np.cos(b0*y) - 1)
			T2_1 = np.sqrt(2*np.pi)*y*np.exp(y**2/2)
			T2_2 = np.real(erf((y+1j*b0)/np.sqrt(2))) - erf(y/np.sqrt(2))

			return (T1 + T2_1*T2_2)
				
		K_data = []

		for entry in self.data:
			a = entry[0]
			phi = entry[1]
			y = np.tan(ampl)*(alpha.real*np.cos(phi)+alpha.imag*np.sin(phi)) - a/np.cos(ampl)
			K_data.append(_Kmod_exp_(y))

		return np.tan(ampl)**2*np.mean(K_data)
	
	# Magnon density Matrix from optical data using MLE. hil_size: Hilbert space of the output, rho_tol: error tolerance in the recursion, max_iter: maximum number of iterations
	def den_from_data_mag(self, hil_size=40, rho_tol=0.01, max_iter=100):
		
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')
		
		data = self.data
	
		print("Computing Projections")
		Projs = [_proj_(entry,self.ampl,self.sq_exp,hil_size) for entry in tqdm.tqdm(data)] 

		print("Fixed point iteration begins")
		return _fixed_pt_(Projs,rho_tol=rho_tol,max_iter=max_iter)




## Helper functions

# Generates data for a given probability density. pmax: The maximum value of probability function, amax: The maximum value of observations, no_samples: number of samples to be generated
def _rejection_sampling_(prob,pmax = 1, amax = 6, no_samples = 1e4): 

	samps_found = 0
	data = []

	while (samps_found < no_samples):
		# Take a random (a,phi)
		a = amax*(2*np.random.rand() - 1)
		phi = np.pi*np.random.rand()

		# Accept (a,phi) with a probability prob(a,phi)/pmax
		p = pmax*np.random.rand()
		if (prob(a,phi) > p):
			data.append([a,phi])
			samps_found += 1

	return data

# The wave-function corresponding to a particular optical observation. This function needs improvements!
def _psi_(entry,hil_size = 40):
	a = entry[0]
	phi = entry[1]
	
	Sq = qt.squeeze(hil_size,1.4*np.exp(2j*phi))  # The squeezing amount 1.5 should depend on the size of hilbert space and displacement. I don't know how to automatize it.
	Disp = qt.displace(hil_size,a*np.exp(1j*phi)/2)

	return Disp*(Sq*qt.fock(hil_size,0)) # The displacement of this state is slightly different than what we want because of the squeezing.

# The projection operator P (as used in the paper) for a given entry = [a,phi], theta = ampl, e^r = sq_exp, and hilbert space size hil_size. We can safely ignore a constant pre-factor as the definition of R doesn't depend on it.
def _proj_(entry,ampl,sq_exp,hil_size):
	a = entry[0]
	phi = entry[1]

	ann = qt.destroy(hil_size)
	quad = np.exp(1j*phi)*ann.dag() + np.exp(-1j*phi)*ann
	exp_arg = -sq_exp**2*(quad*np.sin(ampl) - a)**2/(2*np.cos(ampl)**2)

	return exp_arg.expm()

# Solving the fixed point equation for MLE iteratively with the set of P-operators 'Projs', 'rho_tol' is the tolerance of the iteration, 'max_iter' is the maximum number of iterations allowed.
def _fixed_pt_(Projs,rho_tol = 0.005,max_iter = 100):
	
	Projs_norm = [P/P.tr() for P in Projs if P.tr() > 0]  # To avoid extremely large or extremely small numbers
	N = len(Projs_norm)

	def _R_(rho):
		return sum([P/((rho*P).tr()) for P in Projs_norm])/N # Maybe parallelize this by splitting psi_data into multiple lists

	rho_unnorm = sum(Projs_norm)
	rho_cur = rho_unnorm/rho_unnorm.tr()
	pbar = tqdm.tqdm(total = max_iter)  # The handle for a bar that tracks progress.

	for i in range(max_iter):

		R_cur = _R_(rho_cur)
		rho_next = (R_cur*rho_cur + rho_cur*R_cur)/2
		convergence = (rho_next - rho_cur).norm()

		if (convergence < rho_tol):
			pbar.set_description("Convergence achieved")
			break

		msg_conv = "Error--%.4f" %convergence
		pbar.set_description(msg_conv)
		pbar.update(1)

		rho_cur = rho_next/rho_next.tr()  # Normalize to reduce rounding errors
	
	return rho_cur


