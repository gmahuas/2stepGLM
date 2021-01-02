function bases = gen_basis_stim(data_bases)

UNPACK_STRUCT(data_bases)

% Stimulus filter basis
basis_stim = RCbasis(first_peak_stim, last_peak_stim, ...
    streach_stim, N_basis_stim, N_integ_stim);

bases = PACK_STRUCT('basis_stim');

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