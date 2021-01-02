import numpy as np

def raised_cosine_basis(first_peak, last_peak, streach, n_basis, nt_integ):
    eps = 1e-12
    nl = lambda x : np.log(eps+x)
    inverse_nl = lambda x : np.exp(x)

    nlpos = [nl(first_peak+streach), nl(last_peak+streach)]
    peaks = np.expand_dims(np.linspace(nlpos[0], nlpos[1], n_basis), 1)
    spacing = peaks[1] - peaks[0]
    timepoints = np.expand_dims(np.arange(np.int(inverse_nl(nlpos[1]+5*spacing)-streach)), 0)
    nt = timepoints.shape[0];

    ff = lambda x,c,dc : (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/dc/2)))+1)/2;

    basis = ff(np.tile(nl(timepoints+streach), (n_basis, 1)), np.tile(peaks, (1, nt)), spacing);
    basis = basis[:,:nt_integ]/np.expand_dims(np.sum(basis, axis=1), 1)
    basis = basis[:,::-1]
    return basis

def self_basis_gen(last_peak_self, streach_self, n_basis_self, nt_integ_self, cell, spikes):
    # Estimate refractory period
    isi_temp = []
    n_rep = spikes.shape[-1]
    for repe in range(n_rep):
        spikes_temp = spikes[cell, :, repe]
        isi_temp = isi_temp + list(np.diff([i for i in range(len(spikes_temp)) if spikes_temp[i]==1]))
    isi_temp.sort()
    tau_r = np.int(np.floor(np.median(isi_temp[:10])))
    self_basis = raised_cosine_basis(first_peak=tau_r, last_peak=last_peak_self, streach=streach_self, 
                                     n_basis=n_basis_self, nt_integ=nt_integ_self)
    self_basis[:,-tau_r:] = 0
    return self_basis, tau_r