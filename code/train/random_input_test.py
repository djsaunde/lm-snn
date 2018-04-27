from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os
import pickle
import random
import brian_no_units

import brian as b

from brian             import *
from brian.globalprefs import *
from timeit            import default_timer

set_global_preferences(dt=1.0 * b.ms)

b.log_level_error()


v_rest_e, v_rest_i = -65.0 * b.mV, -60.0 * b.mV
v_reset_e, v_reset_i = -65.0 * b.mV, -45.0 * b.mV
v_thresh_e, v_thresh_i = -52.0 * b.mV, -40.0 * b.mV
refrac_e, refrac_i = 5.0 * b.ms, 2.0 * b.ms

weight, delay = {}, {}

input_population_names = [ 'X' ]
population_names = [ 'A' ]
input_connection_names = [ 'XA' ]
save_conns = [ 'XeAe', 'AeAe' ]

input_conn_names = [ 'ee_input' ]
recurrent_conn_names = [ 'ei', 'ie', 'ee' ]

delay['ee_input'] = (0 * b.ms, 10 * b.ms)

tc_pre_ee, tc_post_ee = 20 * b.ms, 20 * b.ms
nu_ee_pre, nu_ee_post = 0.0001, 0.01
exp_ee_post = exp_ee_pre = 0.2
w_mu_pre, w_mu_post = 0.2, 0.2

tc_theta = 1e7 * b.ms
theta_plus_e = 0.05 * b.mV
scr_e = 'v = v_reset_e; theta += theta_plus_e; timer = 0 * ms'

offset = 20.0 * b.mV
v_thresh_e = '(v>(theta - offset + ' + str(v_thresh_e) + ')) * (timer>refrac_e)'

eqs_e = '''
	dv/dt = ((v_rest_e - v) + (I_synE + I_synI) / nS) / (100 * ms) : volt
	I_synE = ge * nS * -v : amp
	I_synI = gi * nS * (-100. * mV-v) : amp
	dge/dt = -ge / (1.0 * ms) : 1
	dgi/dt = -gi / (2.0 * ms) : 1
	'''
			
eqs_e += '\n  dtheta/dt = -theta / (tc_theta)  : volt'
eqs_e += '\n  dtimer/dt = 100.0 : ms'

eqs_i = '''
	dv/dt = ((v_rest_i - v) + (I_synE + I_synI) / nS) / (10 * ms) : volt
	I_synE = ge * nS * -v : amp
	I_synI = gi * nS * (-85. * mV - v) : amp
	dge/dt = -ge / (1.0 * ms) : 1
	dgi/dt = -gi / (2.0 * ms) : 1
	'''

eqs_stdp = '''
	dpre/dt = -pre / tc_pre_ee : 1.0
	dpost/dt = -post / tc_post_ee : 1.0
	'''

eqs_stdp_pre = 'pre = 1.; w -= nu_ee_pre * post'
eqs_stdp_post = 'w += nu_ee_post * pre; post = 1.'

print('Creating neuron groups.')
groups = {}
groups['e'] = b.NeuronGroup(6400, eqs_e, threshold=v_thresh_e,
	refractory=refrac_e, reset=scr_e, compile=True, freeze=True)
groups['i'] = b.NeuronGroup(6400, eqs_i, threshold=v_thresh_i,
	refractory=refrac_i, reset=v_reset_i, compile=True, freeze=True)

groups['Ae'] = groups['e'].subgroup(6400)
groups['Ai'] = groups['i'].subgroup(6400)

groups['Ae'].v = v_rest_e - 40. * b.mV
groups['Ai'].v = v_rest_i - 40. * b.mV

groups['e'].theta = np.ones(6400) * 20.0 * b.mV

print('Creating connections between excitatory and inhibitory layers.')
connections = {}
connections['Ae_Ai'] = b.Connection(groups['Ae'], groups['Ai'], state='ge')
connections['Ae_Ai'].connect_full(groups['Ae'], groups['Ai'], 10.4)

connections['Ai_Ae'] = b.Connection(groups['Ai'], groups['Ae'], state='gi')
w = 17.4 * np.ones([6400, 6400]) - 17.4 * np.diag(np.ones(6400))
connections['Ai_Ae'].connect(groups['Ai'], groups['Ae'], w)

print('Creating Poisson input group.')
groups['X'] = b.PoissonGroup(784, 0)

print('Creating connection between input and excitatory layer.')
connections['X_Ae'] = b.Connection(groups['X'], groups['Ae'],
				state='ge', delay=True, max_delay=10 * b.ms)
			
w = 0.3 * np.random.rand(784, 6400)
connections['X_Ae'].connect(groups['X'], groups['Ae'], w)

stdp = {}
stdp['X_Ae'] = b.STDP(connections['X_Ae'], eqs=eqs_stdp, pre=eqs_stdp_pre,
										post=eqs_stdp_post, wmin=0, wmax=1)

groups['X'].rate = 0.1 * np.random.rand(784)

print('Running simulation for 10^6 timesteps.')
start = default_timer()
b.run(1000000 * b.ms)
print('Time: %.4f' % (default_timer() - start))

