function bases = gen_basis_couplings(data_bases)

UNPACK_STRUCT(data_bases)

% Coupling filters basis
basis_coupl = RCbasis(first_peak_coupl, last_peak_coupl, ...
    streach_coupl, N_basis_coupl, N_integ_coupl);

% Post spike filter basis
basis_self = RCbasis(first_peak_self, last_peak_self, ...
    streach_self, N_basis_self, N_integ_self);

% Refractory period
[N_neur, ~, N_repe] = size(spikes_train);
tau_r = zeros(N_neur,1);
for neur = 1:N_neur
    isi_temp = [];
    for repe = 1:N_repe
        isi_temp = [isi_temp diff(find(spikes_train(neur,:,repe) == 1))];
    end
    isi_temp = sort(isi_temp);
    tau_r(neur) = floor(median(isi_temp(1:10)));
end

bases = PACK_STRUCT('basis_coupl', 'basis_self', 'tau_r');

end

function basis = RCbasis(PosFirst, PosLast, nlLoc, nBasis, lBasis)

% Non-linearity
nl = @(x)(log(x+eps));
inl = @(x)exp(x);

% Raised Cosine basis
nlPos = nl([PosFirst, PosLast]+nlLoc);  
peaks = linspace(nlPos(1), nlPos(2), nBasis);
spacing = peaks(2)-peaks(1);
timepoints = [0:inl(nlPos(2)+5*spacing)-nlLoc]';
nt = length(timepoints);
ff = @(x,c,dc)(cos(max(-pi,min(pi,(x-c)*pi/dc/2)))+1)/2;
basis = ff(repmat(nl(timepoints+nlLoc), 1, nBasis), repmat(peaks, nt, 1), spacing);
basis = basis(1:lBasis,:)'; basis = basis./sum(basis,2);

end