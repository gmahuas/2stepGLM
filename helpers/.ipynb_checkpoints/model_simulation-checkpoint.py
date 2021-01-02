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
    
    # In order to have n_bins_psth pair (so that we can easily get the center
    # bin corresponding to the zero time lag noise covariance
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
    # Mean spike count across time and repeats
    mean_lbd = np.mean(lbd, 1)
    
    # Stimulus covariance
    z_lbd = lbd - np.transpose(np.tile(mean_lbd, (n_bins_psth,1)))
    cov_stim = np.zeros((n_cells, n_cells, 2*max_corr_time+1))
    for tau in np.arange(-max_corr_time,max_corr_time+1):
        cov_stim[:,:,max_corr_time+tau] = np.matmul(z_lbd,np.transpose(np.roll(z_lbd, tau, axis=1)))/n_bins_psth
        
    # Total covariance
    z_n_spikes = n_spikes - np.transpose(np.tile(mean_lbd, (n_rep, n_bins_psth, 1)))
    cov_tot = np.zeros((n_cells, n_cells, 2*max_corr_time+1))
    cov_tot_temp = np.zeros((n_cells, n_cells, n_rep))
    for tau in np.arange(-max_corr_time, max_corr_time+1):
        for rep in np.arange(n_rep):
            cov_tot_temp[:,:,rep] = np.matmul(z_n_spikes[:,:,rep], np.transpose(np.roll(z_n_spikes[:,:,rep], tau, axis=1)))/n_bins_psth
        cov_tot[:,:,max_corr_time+tau] = np.mean(cov_tot_temp, 2)
        
    # Noise covariance
    l_n = n_spikes - lbd[:,:,np.newaxis]
    cov_noise = np.zeros((n_cells, n_cells, 2*max_corr_time+1))
    l_n_l_n = np.zeros((n_cells, n_cells, n_rep))
    
    for tau in np.arange(-max_corr_time, max_corr_time+1):
        l_n_shift = np.roll(l_n, tau, 1)
        for rep in range(n_rep):
            l_n_l_n[:,:,rep] = np.matmul(l_n[:,:,rep], np.transpose(l_n_shift[:,:,rep]))/n_bins_psth
        cov_noise[:,:,max_corr_time+tau] = np.mean(l_n_l_n,2)
    
    # Correlations
    corr_noise = []
    corr_stim = []
    for i_cell in range(n_cells-1):
        for j_cell in np.arange(i_cell+1,n_cells):
            corr_noise += [cov_noise[i_cell,j_cell,max_corr_time]/np.sqrt(cov_tot[i_cell,i_cell,max_corr_time]*cov_tot[j_cell,j_cell,max_corr_time])]
            corr_stim += [cov_stim[i_cell,j_cell,max_corr_time]/np.sqrt(cov_tot[i_cell,i_cell,max_corr_time]*cov_tot[j_cell,j_cell,max_corr_time])]
           
    return corr_noise, corr_stim