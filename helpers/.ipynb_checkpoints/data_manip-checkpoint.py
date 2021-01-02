import numpy as np
import tensorflow as tf
import pickle5 as pkl
import os

# Project the data on the temporal bases
def project_time_series(time_series, basis):
    
    # some params
    n_dims, nt, n_rep = time_series.shape
    n_basis, nt_integ = basis.shape
    
    # reshape basis for use with conv2d
    basis = np.expand_dims(np.float32(basis), [1, 3, 4])
    
    # time_series
    ts = np.reshape(time_series, [n_dims, nt*n_rep], order='F' )
    ts = np.expand_dims(ts, [0, 3])
    ts = tf.cast(ts[:,:,:,:], dtype='float32')
    ts_train = np.zeros((n_rep*(nt-nt_integ), n_dims, n_basis), dtype='float32')
    for ibasis in range(n_basis):
        ts_temp = tf.nn.conv2d(ts, basis[ibasis,:], strides=[1,1,1,1], padding='VALID')
        ts_temp = tf.concat((tf.zeros((1, n_dims, nt_integ, 1)), ts_temp[:,:,:-1,:]), axis=2)
        ts_temp = tf.reshape(ts_temp, [n_dims, n_rep, nt])
        ts_temp = tf.reshape(ts_temp[:,:,nt_integ:], [n_dims, (nt-nt_integ)*n_rep])
        ts_train[:,:,ibasis] = tf.transpose(ts_temp)
    ts_temp = tf.reshape(ts_train, [(nt-nt_integ)*n_rep, n_dims*n_basis])
    return ts_temp

def project_sliced_dataset(stimulus, spikes, cell, ipart, n_parts_dataset, stim_basis, model, coupl_basis, self_basis, tau_r):

    # some declarations
    nxy, nt, n_rep = stimulus.shape
    n_cells = spikes.shape[0]
    n_basis_stim, nt_integ_stim = stim_basis.shape
    output = []
    
    # compute size of the slice
    nt_part = int(np.ceil((nt-nt_integ_stim)/n_parts_dataset))
    nt_curr_part = int(min((ipart+1)*nt_part, nt-nt_integ_stim))-ipart*nt_part
    
    # project stimulus on temporal basis
    stim_slice = stimulus[:,ipart*nt_part:np.uint(min((ipart+1)*nt_part+nt_integ_stim, nt)),:]-0.5
    output.append(project_time_series(stim_slice, stim_basis))
    
    # project spikes
    if model=='GLM':
        # coupling basis
        n_basis_coupl, nt_integ_coupl = coupl_basis.shape
        spikes_slice = spikes[np.setdiff1d(range(n_cells),cell),ipart*nt_part + 
                              (nt_integ_stim-nt_integ_coupl):np.uint(min((ipart+1)*nt_part+nt_integ_stim, nt)),:]
        output.append(project_time_series(spikes_slice, coupl_basis))
        # self basis
        n_basis_self, nt_integ_self = self_basis.shape
        spikes_slice = spikes[[cell],ipart*nt_part + 
                              (nt_integ_stim-nt_integ_self):np.uint(min((ipart+1)*nt_part+nt_integ_stim, nt)),:]
        output.append(project_time_series(spikes_slice, self_basis))
        # refractory period
        refr_fn = np.ones((1,tau_r)) 
        spikes_slice = spikes[[cell],ipart*nt_part + 
                              (nt_integ_stim-tau_r):np.uint(min((ipart+1)*nt_part+nt_integ_stim, nt)),:]
        output.append(project_time_series(spikes_slice, refr_fn))
        # spike train
        output.append(tf.cast(np.reshape(spikes[cell,nt_integ_stim +ipart*nt_curr_part:nt_integ_stim+
                             (ipart+1)*nt_curr_part,:], [nt_curr_part*n_rep], order='F'), dtype='float32'))
    else:
        # For the LN model, no need to recompute the dataset for each cell
        output.append(tf.cast(np.reshape(spikes[:,nt_integ_stim +ipart*nt_curr_part:nt_integ_stim+
                             (ipart+1)*nt_curr_part,:], [n_cells,nt_curr_part*n_rep], order='F'), dtype='float32'))
        
    return tuple(output)

def project_testset(stimulus, spikes, cell, stim_basis, model, coupl_basis, self_basis, tau_r):

    nt_integ_stim = stim_basis.shape[1]
    n_cells = spikes.shape[0]
    output = []
    
    # stimulus
    stimulus = np.expand_dims(stimulus, 2)-0.5
    output.append(project_time_series(stimulus, stim_basis))
    
    # project spikes
    if model=='GLM':
        # coupling basis
        nt_integ_coupl = coupl_basis.shape[1 ]
        spikes_slice = spikes[np.setdiff1d(range(n_cells),cell),nt_integ_stim-nt_integ_coupl:,:]
        output.append(project_time_series(spikes_slice, coupl_basis))
        # self basis
        nt_integ_self = self_basis.shape[1]
        spikes_slice = spikes[[cell],nt_integ_stim-nt_integ_self:,:]
        output.append(project_time_series(spikes_slice, self_basis))
        # refractory period
        refr_fn = np.ones((1,tau_r))
        spikes_slice = spikes[[cell],nt_integ_stim-tau_r:,:]
        output.append(project_time_series(spikes_slice, refr_fn))

    return tuple(output)
    
# Build and load the dataset
def build_dataset(stimulus, spikes, max_size_dataslice, stim_basis, model, cell=None,
               coupl_basis=None, self_basis=None, tau_r=None):
    
    if model=='GLM' and (coupl_basis.any() == None or self_basis.any() == None or tau_r == None):
        raise Exception('Please provide the coupling bases and the refractory period')
    elif model!='LN' and model!='GLM':
        raise Exception('Specify model type: LN or GLM')
    
    # Some parameters
    nxy, nt, n_rep = stimulus.shape
    n_cells, *dull = spikes.shape
    n_basis_stim, nt_integ_stim = stim_basis.shape
    if model=='GLM':
        n_basis_coupl = coupl_basis.shape[0]
        n_basis_self = self_basis.shape[0]
    else:
        n_basis_coupl, n_basis_self = 0, 0
    
    total_size = ((nt-nt_integ_stim)*n_rep*(nxy*n_basis_stim + 1) + 
                  (nt-nt_integ_stim)*n_rep*((n_cells-1)*n_basis_coupl+n_basis_self))*4/1024**2 # in MB
    print(f'Total size of the dataset: {total_size}')
    n_parts_dataset = int(np.ceil(total_size/max_size_dataslice))
    print(f'Split in {n_parts_dataset} parts')
    
    # Save the dataset slices on the disk
    # Create data directory
    if not os.path.exists('dataset_temp'):
        os.mkdir('dataset_temp')
    # Split and save the dataset
    for ipart in range(n_parts_dataset):
        sliced_data = project_sliced_dataset(stimulus, spikes, cell, ipart, n_parts_dataset, stim_basis, 
                                              model, coupl_basis, self_basis, tau_r)
        print(f'part {ipart+1} length: {sliced_data[-1].shape[-1]}')
        with open('./dataset_temp/dataset'+str(ipart)+'.temp', 'wb') as f:
            pkl.dump(sliced_data, f)
    
    return n_parts_dataset
            
def load_dataset(model, cell, batch_size, ipart, n_basis_coupl=None, n_basis_self=None):
    
    with open('./dataset_temp/dataset'+str(ipart)+'.temp', 'rb') as f:
        data_slice = pkl.load(f)
    
    # Extract data
    stim_data = data_slice[0]
    spikes_data = data_slice[-1]
    if model == 'GLM':
        if n_basis_coupl == None or n_basis_self == None:
            raise Exception('Provide number of coupling and self basis elements')
        else:
            coupl_data = data_slice[1]
            self_data = data_slice[2]
            refr_data = data_slice[3]
        dataset_slice = tf.data.Dataset.from_tensor_slices((stim_data, coupl_data, self_data, refr_data, spikes_data))
        dataset_slice = dataset_slice.shuffle(spikes_data.shape[0]).batch(batch_size)
    elif model == 'LN':
        dataset_slice = tf.data.Dataset.from_tensor_slices((stim_data, spikes_data[cell,:]))
        dataset_slice = dataset_slice.shuffle(spikes_data.shape[0]).batch(batch_size)
    else:
        raise Exception('Specify model type: LN or GLM')
    
    return dataset_slice

def build_interaction_dataset(spikes, cell, coupl_basis, self_basis, tau_r):

    # some declarations
    n_cells, nt, n_rep = spikes.shape
    nt_integ_self = self_basis.shape[1]
    nt_integ_coupl = coupl_basis.shape[0]
    nt_integ = np.max([nt_integ_self, nt_integ_coupl])
    output = []
    
    # compute size of the slice
    nt_part = nt-nt_integ
    
    # stimulus
    output.append(tf.cast(np.tile(np.arange(nt_part), n_rep), dtype='float32'))
    
    # coupling basis
    n_basis_coupl, nt_integ_coupl = coupl_basis.shape
    spikes_slice = spikes[np.setdiff1d(range(n_cells),cell),nt_integ-nt_integ_coupl:,:]
    output.append(project_time_series(spikes_slice, coupl_basis))
    # self basis
    n_basis_self, nt_integ_self = self_basis.shape
    spikes_slice = spikes[[cell],nt_integ-nt_integ_self:,:]
    output.append(project_time_series(spikes_slice, self_basis))
    # refractory period
    refr_fn = np.ones((1,tau_r)) 
    spikes_slice = spikes[[cell],nt_integ-tau_r:,:]
    output.append(project_time_series(spikes_slice, refr_fn))
    
    # spikes train
    output.append(tf.cast(np.reshape(spikes[cell,nt_integ:,:], [(nt-nt_integ)*n_rep], order='F'), dtype='float32'))
    
    return tuple(output)