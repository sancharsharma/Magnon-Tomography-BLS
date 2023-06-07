import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import qutip as qt
import tqdm
import tqdm.auto
import h5py
from os.path import exists

from importlib import reload
import Classes as Cls
reload(Cls)
import Sys_defs as Def
reload(Def)


Nm = 40 
sqexp_obs = 5 # 'exp(r)' in paper's notation
no_samples = 1e4 # No of 'experimental' data points
rho_tol=0.01 # Error tolerance for convergence

rand = lambda min,max: (max - min)*np.random.rand() + min # Just a convenient random generator

#file_name = 'Coh_woSq.hf'  # File where data will be stored
#alpha = 1.8 - 2.4j # rand(2,3)*np.exp(1j*rand(-np.pi,np.pi))
#sqexp = 1.45 # np.exp(rand(0,0.7))
#sqdir = 2.67 # rand(0,np.pi)
#mag_state = qt.displace(Nm,alpha)*qt.squeeze(Nm,np.log(sqexp)*np.exp(2j*sqdir))*qt.fock(Nm,0)

file_name = 'Cat_WithSq.hf'
alpha = 2.7 + 1.3j # rand(2,3)*np.exp(1j*rand(-np.pi,np.pi))
phase = -1.73
mag_state = ( (qt.displace(Nm,alpha) + np.exp(1j*phase)*qt.displace(Nm,-alpha))*qt.fock(Nm,0) ).unit()

ampl_list = np.array([0.02,0.25,0.45])*np.pi # 'theta' in paper's notation
results = []

for ampl in ampl_list:
	print("Starting...") 
	#Sys = Def.sqcoh_magnon(alpha,sqexp,sqdir,no_samples=no_samples,hil_size=Nm,ampl=ampl,sq_exp=sqexp_obs) ## Squeezed Coherent State
	Sys = Def.cat_magnon(alpha,phase,no_samples=no_samples,hil_size=Nm,ampl=ampl,sq_exp=sqexp_obs) ## Cat State

	rho_found = Sys.den_from_data_mag(hil_size=Nm,rho_tol=rho_tol)

	fid = qt.fidelity(mag_state,rho_found)
	
	results.append([Sys,rho_found,fid])
	del Sys


def _get_wigner_(rho, vec):

	# Our convention W_{mag} is different than qutip's convention W_Q: W_{mag}(alpha_R,alpha_I) = 2\pi W_Q(\sqrt{2}*alpha_R,sqrt(2)*alpha_I)
	
	qutip_arg = np.sqrt(2)*vec
	W_qutip = qt.wigner(rho, qutip_arg, qutip_arg, method='iterative')
	W = 2*np.pi*W_qutip

	return W

file = h5py.File(file_name,'w')  # Clears out the content if the file already exists

### Parameters
pars = file.create_group('parameters')
pars.create_dataset('no_samples',data=no_samples)
pars.create_dataset('alpha',data=alpha)
pars.create_dataset('phase',data=phase)
	
### Wigner axis
alpha_max = 6
vec = np.linspace(-alpha_max, alpha_max, 500)
pars.create_dataset('wigner_args',data=vec)

### Target
tar = file.create_group('target')
tar_wigner = _get_wigner_(mag_state,vec)
tar.create_dataset('wigner',data=tar_wigner)
tar.create_dataset('wavefunc',data=np.array(mag_state))

grp_names = ['low_SNR','med_SNR','high_SNR'] 
for i in range(3):
	grp = file.create_group(grp_names[i])
	den = results[i][1]
	fid = results[i][2]
	wig = _get_wigner_(den,vec)

	grp.create_dataset('wigner',data=wig)
	grp.create_dataset('den_mat',data=np.array(den))
	grp.create_dataset('fidelity',data=fid)
	grp.create_dataset('ampl',data=ampl_list[i])

	del grp

file.close()
