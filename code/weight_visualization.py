import numpy as np
from pylab import *
import matplotlib.cm as cm

chosenCmap = cm.get_cmap('hot_r') #cm.get_cmap('gist_ncar')

# number of excitatory neurons
n_e_input = raw_input('Enter number of excitatory / inhibitory neurons (default 100): ')
if n_e_input == '':
    n_e = 100
else:
    n_e = int(n_e_input)

# number of inhibitory neurons
n_i = n_e

# set ending of filename saves
ending = str(n_e)

# determine STDP rule to use
stdp_input = ''

if raw_input('Use weight dependence (default no)?: ') in [ 'no', '' ]:
    use_weight_dependence = False
    stdp_input += 'no_weight_dependence_'
else:
    use_weight_dependence = True
    stdp_input += 'weight_dependence_'

if raw_input('Enter (yes / no) for post-pre (default yes): ') in [ 'yes', '' ]:
    post_pre = True
    stdp_input += 'postpre'
else:
    post_pre = False
    stdp_input += 'no_postpre'

readout_name = 'XeAe' + str(n_e) + '_' + stdp_input

def computePopVector(popArray):
    size = len(popArray)
    complex_unit_roots = np.array([np.exp(1j*(2*np.pi/size)*cur_pos) for cur_pos in xrange(size)])
    cur_pos = (np.angle(np.sum(popArray * complex_unit_roots)) % (2*np.pi)) / (2*np.pi)
    return cur_pos

def get_2d_input_weights():
    weight_matrix = XA_values
    n_e_sqrt = int(np.sqrt(n_e))
    n_in_sqrt = int(np.sqrt(n_input))
    num_values_col = n_e_sqrt*n_in_sqrt
    num_values_row = num_values_col
    rearranged_weights = np.zeros((num_values_col, num_values_row))
        
    for i in xrange(n_e_sqrt):
        for j in xrange(n_e_sqrt):
                rearranged_weights[i*n_in_sqrt : (i+1)*n_in_sqrt, j*n_in_sqrt : (j+1)*n_in_sqrt] = \
                    weight_matrix[:, i + j*n_e_sqrt].reshape((n_in_sqrt, n_in_sqrt))
    return rearranged_weights

def plot_2d_input_weights():
    name = 'XeAe'
    weights = get_2d_input_weights()
    fig = figure(figsize = (18, 18))
    im2 = imshow(weights, interpolation = "nearest", vmin = 0, cmap = chosenCmap) #my_cmap
    colorbar(im2)
    title('weights of connection' + name)
    fig.canvas.draw()
    return im2, fig


bright_grey = '#f4f4f4'
red   = '#ff0000'
green   = '#00ff00'
black   = '#000000'
my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('own2',[bright_grey,black])

n_input = 784


readout = np.load('../weights/eth_model_weights/' + readout_name + '.npy')
value_arr = np.nan * np.ones((n_input, n_e))
connection_parameters = readout

for conn in connection_parameters: 
    src, tgt, value = conn
    if np.isnan(value_arr[src, tgt]):
        value_arr[src, tgt] = value
    else:
        value_arr[src, tgt] += value

for i in xrange(n_e):
    print values[i,i]
    
fi = figure()
im = imshow(values, interpolation="nearest", cmap = chosenCmap, aspect='auto')  # copper_r   autumn_r  Greys  my_cmap  gist_rainbow
cbar = colorbar(im)

xlabel('Target excitatory neuron number')
ylabel('Source excitatory neuron number')

title(name)
savefig(str(fi.number))

XA_values = np.copy(values)
    
im, fi = plot_2d_input_weights()
savefig(str(fi.number))

XA_sum = np.nansum(XA_values[0:n_input,0:n_e], axis = 0)/n_e
AA_sum = np.nansum(AA_values[0:n_e,0:n_e], axis = 0)/n_e

fi = figure()
plot(XA_sum, AA_sum, 'w.')
for label, x, y in zip(range(200), XA_sum, AA_sum):
    plt.annotate(label, 
                xy = (x, y), xytext = (-0, 0),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                color = 'k')

xlabel('summed input from X to A for A neurons')
ylabel('summed input from A to A for A neurons')
savefig(str(fi.number))

show()
