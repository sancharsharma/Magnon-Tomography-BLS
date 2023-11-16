import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py

from importlib import reload

# This expects a .hf file and a folder inside Figures with the same name <category>
category = 'Cat_WithSq'
data_file = category + '.hf'
folder = 'Figures/' + category + '/'

# Generates the contour plot with x and y axes being the vector args and the color values given by the vector 'vals'
def _contour_plot_(vals, args, fig=None, ax=None):

	if not fig and not ax:
		fig, ax = plt.subplots(1, 1)

	wlim = abs(vals).max()

	cmap = mpl.cm.get_cmap('RdBu')  # A color map

	cf = ax.contourf(args, args, vals, 300,
                         norm=mpl.colors.Normalize(-wlim, wlim), cmap=cmap)  # The plot

	near_max = int(round(0.9*args.max()))
	axis_ticks = [-near_max,0,near_max]  # The minimal number of ticks
	ax.set_xticks(axis_ticks)
	ax.set_yticks(axis_ticks)

	# tickpos is vector of ticks on the colorbar
	if (vals.min() > -0.05*vals.max()): # If the minimum is too close to 0, we don't need to put it on the colorbar
		tickpos = [0.95*vals.max(),0]
	else:
		tickpos = [0.95*vals.max(),0,0.95*vals.min()]

	cbar = fig.colorbar(cf, ax=ax, ticks=tickpos)
	cbar.ax.set_yticklabels(['{v:.2f}'.format(v=tp) for tp in tickpos])

	return fig, ax

##### Code to save every figure separately in a file. It requires a folder named Figures in the directory of working.

plt.rcParams.update({'font.size': 22})
grp_names = ['low_SNR','med_SNR','high_SNR']  # These are the group names in the h5 file to be read

data = h5py.File(data_file,'r')
args = np.array(data['parameters/wigner_args'])  # The range of Wigner arguments
tar_wig = np.array(data['target/wigner'])  # The target Wigner function data
recons_wigs = [np.array(data[name + '/wigner']) for name in grp_names]  # The reconstructed state Wigner function data
data.close()

fig_tar, ax_tar = _contour_plot_(tar_wig,args) 
fig_tar.savefig(folder + 'target.png',dpi = 400,bbox_inches = 'tight')

for i in range(3):

	name = grp_names[i]
	fig, ax = _contour_plot_(recons_wigs[i],args)
	fig.savefig(folder + name + '.png',dpi = 400,bbox_inches = 'tight')

	del fig,ax


