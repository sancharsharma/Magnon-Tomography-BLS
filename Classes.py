import scipy.integrate as integ
from scipy.special import erf
import numpy as np
from math import factorial
import multiprocess as mp
import qutip as qt
import tqdm


class Tomo:
	
	def __init__(self):
		self.data = 'none'
		self.ph_state = 'none'
		self.mag_state = 'none'
		self.ph_wig = 'none'
		self.mag_wig = 'none'
		self.ph_prob = 'none'
		self.mag_prob = 'none'
		self.ampl = 'none'
		self.sq_exp = 1

	##################### Photon Functions
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
	
	def data_from_prob_ph(self,pmax = 1, amax = 6, no_samples = 5e3):
		# Using the so-called rejection sampling

		if (self.ph_prob == 'none'):
			raise ValueError('Supply photon probability function using self.ph_prob = lambda a,phi: ...')
		
		return _rejection_sampling_(self.ph_prob,pmax = pmax, amax = amax, no_samples = no_samples)
		
		
	def wig_from_data_ph(self,alpha,kc=5): #alpha wigner function input and kc as cut-off in the kernel
		
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
	
	def den_from_data_ph(self, hil_size=40, rho_tol=0.005, max_iter=100): #hil_size: Hilbert space of the output, rho_tol: error tolerance in the recursion, max_iter: maximum number of iterations
		
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')
		
		data = self.data

		print("Computing Projections")
		psi_data = [_psi_(entry,hil_size = hil_size) for entry in tqdm.tqdm(data)]
		Projs = [psi*psi.dag() for psi in psi_data]
		

		print("Fixed point iteration begins")
		return _fixed_pt_(Projs,rho_tol=rho_tol,max_iter=max_iter)


	##################### Magnon Functions
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


	def wig_from_data_mag(self,alpha,b0=2):
		
		print("This generally gives terrible results. Try your own regularization.")
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')

		if (self.ampl == 'none'):
			raise ValueError('Supply ampl using self.ampl = ...]')

		ampl = self.ampl

		def _Kmod_exp_(y): # In the notation of the paper, this is Kmod(y*cos(ampl))/tan(ampl)**2
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
	
	def den_from_data_mag(self, hil_size=40, rho_tol=0.01, max_iter=100):
		
		if (self.data == 'none'):
			raise ValueError('Supply data using self.data = [[a1,phi1],...]')
		
		data = self.data
	
		print("Computing Projections")
		Projs = [_proj_(entry,self.ampl,self.sq_exp,hil_size) for entry in tqdm.tqdm(data)] 

		print("Fixed point iteration begins")
		return _fixed_pt_(Projs,rho_tol=rho_tol,max_iter=max_iter)




## Helper functions

def _rejection_sampling_(prob,pmax = 1, amax = 6, no_samples = 5e3): # pmax: The maximum value of probability function, amax: The maximum value of observations, no_samples: number of samples to be generated

	samps_found = 0
	data = []

	while (samps_found < no_samples):
		# Take a random (a,phi)
		a = amax*(2*np.random.rand() - 1)
		phi = np.pi*np.random.rand()

		# Reject (a,phi) with a relative probability prob(a,phi)
		p = pmax*np.random.rand()
		if (prob(a,phi) > p):
			data.append([a,phi])
			samps_found += 1

	return data

def _psi_(entry,hil_size = 40):
	a = entry[0]
	phi = entry[1]
	
	Sq = qt.squeeze(hil_size,1.4*np.exp(2j*phi))  # The squeezing amount should depend on the size of hilbert space and displacement.
	Disp = qt.displace(hil_size,a*np.exp(1j*phi)/2)
	return Disp*(Sq*qt.fock(hil_size,0)) # The displacement of this state is slightly different than what we want.


def _proj_(entry,ampl,sq_exp,hil_size):
	a = entry[0]
	phi = entry[1]

	ann = qt.destroy(hil_size)
	quad = np.exp(1j*phi)*ann.dag() + np.exp(-1j*phi)*ann
	exp_arg = -sq_exp**2*(quad*np.sin(ampl) - a)**2/(2*np.cos(ampl)**2)

	return exp_arg.expm()

def _fixed_pt_(Projs,rho_tol = 0.005,max_iter = 100):
	
	Projs_norm = [P/P.tr() for P in Projs if P.tr() > 0]
	N = len(Projs_norm)

	def _R_(rho):
		return sum([P/((rho*P).tr()) for P in Projs_norm])/N # Maybe parallelize this by splitting psi_data into multiple lists

	rho_unnorm = sum(Projs_norm)
	rho_cur = rho_unnorm/rho_unnorm.tr()
	pbar = tqdm.tqdm(total = max_iter)

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

		rho_cur = rho_next/rho_next.tr()

	
	return rho_cur




	
