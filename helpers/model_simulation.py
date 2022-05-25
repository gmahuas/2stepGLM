import numpy as np

def int_potential(spikes_past, coupling_filters_list, self_filter_list, tau_r_list):
    n_cells = spikes_past.shape[0]
    h_int = np.zeros(n_cells)
    for cell in range(n_cells):
        other_cells = np.setdiff1d(np.arange(n_cells), cell)
        coupling_filters = coupling_filters_list[cell]
        self_filter = self_filter_list[cell]
        
        h_coupl = np.sum(spikes_past[other_cells, :]*coupling_filters)
        h_self = np.sum(spikes_past[cell, :]*self_filter)
        
        h_int[cell] = h_coupl + h_self
    return h_int

def interaction_potential_2steps(h_past, spikes_past, coupling_filters_list, self_filter_list, tau_r_list):
    n_cells = spikes_past.shape[0]
    h_int = np.zeros(n_cells)
    for cell in range(n_cells):
        other_cells = np.setdiff1d(np.arange(n_cells), cell)
        coupling_filters = coupling_filters_list[cell]
        self_filter = self_filter_list[cell]
        
        h_coupl = np.sum((spikes_past[other_cells, :]-np.exp(h_past[other_cells,:]))*coupling_filters)
        h_self = np.sum((spikes_past[cell, :]-np.exp(h_past[cell,:]))*self_filter)

        # correction for the addition of the mean refractory period
        refr_corr = np.sum(np.exp(h_past[cell,-tau_r_list[cell]:]))

        h_int[cell] = h_coupl + h_self + refr_corr
    return h_int

def compute_correlations(spikes, bin_size, max_spikes, max_corr_time=10):
    n_cells, nt, n_rep = spikes.shape
    
    # Keep only the good repetitions
    good_rep_list = []
    
    for cell in range(n_cells):
        sat_rep = []
        for rep in range(n_rep):
            if np.sum(spikes[cell,:,rep]) > max_spikes:
                sat_rep += [rep]
        good_rep = np.setdiff1d(np.arange(n_rep), sat_rep)
        good_rep_list += [good_rep]
        
    sat_list = []
    for cell in range(n_cells):
        sat_list += list(np.setdiff1d(np.arange(n_rep), good_rep_list[cell]))
    
    sat_list = np.unique(sat_list)

    good_list = np.setdiff1d(np.arange(n_rep), sat_list)
    
    if sat_list.size:
        print(f'Repetition(s) {sat_list} were discarded, had more than {max_spikes} spikes')
        
    n_rep = good_list.shape[0]
    spikes = spikes[:,:,good_list]
    
    # Bin the spike response
    n_bins_psth = np.floor(nt/bin_size)
    tronc = nt%bin_size
    
    if n_bins_psth%2 == 1:
        n_bins_psth += -1
        tronc += bin_size
    n_bins_psth = n_bins_psth.astype(int)
        
    spikes = np.reshape(spikes, [n_cells, nt, n_rep], order='F')
    if tronc != 0:
        spikes = np.reshape(spikes[:,:-tronc,:], [n_cells, (nt-tronc)*n_rep], order='F')
    else:
        spikes = np.reshape(spikes, [n_cells, nt*n_rep], order='F')
    
    n_spikes = np.squeeze(np.sum(np.reshape(spikes, [n_cells, bin_size, n_bins_psth*n_rep], order='F'), axis=1))
    n_spikes = np.reshape(n_spikes, [n_cells, n_bins_psth, n_rep], order='F')
    
    # Covariances
    # Firing rates
    lbd = np.mean(n_spikes, 2)

    # Covariances
    cov_stim = np.cov(lbd, ddof=0)
    cov_tot = np.cov(np.reshape(n_spikes, (n_cells, n_bins_psth*n_rep), order='F'), ddof=0)
    cov_noise = cov_tot - cov_stim
    
    var_tot = np.diag(cov_tot)
    
    # Correlations
    corr_noise_array = (var_tot**(-0.5)*np.eye(n_cells))@cov_noise@(var_tot**(-0.5)*np.eye(n_cells))
    corr_stim_array = (var_tot**(-0.5)*np.eye(n_cells))@cov_stim@(var_tot**(-0.5)*np.eye(n_cells))
    
    # Correlations list
    corr_noise = []
    corr_stim = []
    for i_cell in range(n_cells-1):
        for j_cell in np.arange(i_cell+1,n_cells):
            corr_noise += [corr_noise_array[i_cell,j_cell]]
            corr_stim += [corr_stim_array[i_cell,j_cell]]

    return corr_noise, corr_stim
