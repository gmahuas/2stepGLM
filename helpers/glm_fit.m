function filters = glm_fit(data_fit)
%% Unpack data
UNPACK_STRUCT(data_fit)
[N_neur,~,N_repe] = size(spikes);
[Nx, Ny, ~, ~] = size(stimulus);

%% Some helper matrices for the computations
% For the stimulus potential
data_conv_stim = PACK_STRUCT('stimulus', 'Nx', 'Ny', 'N_integ_stim', 'N_repe', ...
    'Nt_inf', 'N_basis_stim', 'basis_stim', 'dt');
conv_mat_stim = conv_stim(data_conv_stim);

% For the laplacian
if Ny > 1 % if 2D stimulus
    lapl_mat = zeros(Nx*Ny, Nx*Ny, 2);
    lapl_x=full(gallery('tridiag',Nx,1,-2,1));
    lapl_y=full(gallery('tridiag',Ny,1,-2,1));
    lapl_x(1,end) = 1;
    lapl_x(end,1) = 1;
    lapl_y(1,end) = 1;
    lapl_y(end,1) = 1;

    lapl_mat_x = lapl_x;
    for i = 1:Ny-1
        lapl_mat_x = blkdiag(lapl_mat_x, lapl_x);
    end
    
    lapl_mat_y = lapl_y;
    for i = 1:Ny-1
        lapl_mat_y = blkdiag(lapl_mat_y, lapl_y);
    end
    lapl_mat(:,:,1) = lapl_mat_x;
    lapl_mat(:,:,2) = lapl_mat_y;
elseif Ny == 1 % : 1D stimulus
    lapl_mat=full(gallery('tridiag',Nx,1,-2,1));
    lapl_mat(1,end) = 1;
    lapl_mat(end,1) = 1;
end

%% Inference

% Fminunc options
opts = optimset('Gradobj','on','Hessian','off','display','iter', ...
    'MaxIter', 50000, 'MaxFunEvals', 50000, 'TolFun', 1e-6, 'TolX', 1e-6, ...
    'DerivativeCheck', 'off');

for icell = cells
    disp(['Cell #' num2str(icell)])
    
    % L1 on the couplings
    L1_int = ones(N_neur,1)*L1_coupl;
    % L1 on the self
    L1_int(icell) = L1_self;
    
    % Spike train of the studied cell
    spikes_cell = reshape(squeeze(spikes(icell,N_integ+1:end,:)),N_repe*Nt_inf,1);

    % Initial guess for the parameters
    stim_coeff = squeeze(stim_coeff_list(icell,:,:));
    offset = offset_list(icell);
    coup_coeff = squeeze(coup_coeff_list(icell,:,:));
    self_coeff = squeeze(self_coeff_list(icell,:));

    % Packing the parameters in vector
    params = [];
    
    % Couplings adn self coefficients
    if infer_couplings
        for i = 1:N_neur
            if i == icell
                if infer_self
                    % !!! The first parameter is fixed to refrac_self
                    params = [params self_coeff(2:end)];
                    refr_period = self_coeff(1);
                else
                    refr_period = [];
                end
            else
                params = [params coup_coeff(:,i)'];
            end
        end
    else
        if infer_self
            params = [params self_coeff(2:end)];
            refr_period = self_coeff(1);
        else
            refr_period = [];
        end
    end
    
    % Stimulus coefficients
    params = [params reshape(stim_coeff', 1, N_basis_stim*Nx*Ny)];
    % Offset
    params = [params offset];

    % Useful matrix for the computation of the interaction effects
    if infer_couplings || infer_self
        data_conv_int = PACK_STRUCT('icell', 'spikes', 'dt', 'Nt_inf', ...
            'basis_coupl', 'N_basis_coupl', 'N_integ_coupl', ...
            'basis_self', 'N_basis_self', 'N_integ_self', ...
            'refr_period', 'N_integ', 'N_integ_stim');
        conv_mat_int = conv_int(data_conv_int);
    else 
        conv_mat_int = [];
    end
    
    % Keep only usefull stuff
    if ~infer_couplings && infer_self
        conv_mat_int = conv_mat_int(icell,:,:);
        L1_int = L1_int(icell,:);
    end
    
    % Minimization of the log-likelihood
    data_llk = PACK_STRUCT('icell', 'N_basis_coupl', 'N_basis_self', ...
        'N_basis_stim', 'N_neur', 'Nx', 'Ny', 'conv_mat_stim', ...
        'conv_mat_int', 'spikes_cell', 'L1_int', 'L1_stim', 'L2_lapl_stim', ...
        'infer_couplings', 'lapl_mat', 'dt', 'infer_self', 'infer_couplings');

    params = fminunc(@(x)llk(x, data_llk), params, opts);

    % Recover the parameters
    if infer_couplings
        for i = 1:N_neur
            if i == icell
                if infer_self
                    % ! First coeff is fixed to refrac_self
                    self_coeff_list(icell,2:end) = params(1:N_basis_self);
                    params(1:N_basis_self) = [];
                end
            else
                coup_coeff_list(icell,:,i) = params(1:N_basis_coupl);
                params(1:N_basis_coupl) = [];
            end
        end
    else
        if infer_self
            self_coeff_list(icell,2:end) = params(1:N_basis_self);
            params(1:N_basis_self) = [];
        end
    end
    stim_coeff_list(icell,:,:) = reshape(params(1:Nx*Ny*N_basis_stim), N_basis_stim, Nx*Ny)';
    offset_list(icell) = params(end);           
end

%% We reconstruct the filters

% Stimulus filters
stim_filters_list = zeros(Nx, Ny, N_integ_stim, N_neur);
for i = 1:N_neur
    stim_filter = squeeze(stim_coeff_list(i,:,:))*basis_stim;
    stim_filters_list(:,:,:,i) = reshape(stim_filter, Nx, Ny, N_integ_stim);
end

% Interactions and self
int_filters_list = zeros(N_neur,max(N_integ_coupl, N_integ_self), N_neur);
for i = 1:N_neur
    for j = 1:N_neur
        % filter from j to i
        if j == i
            if infer_self
                int_filter = self_coeff_list(i,2:end)*basis_self;
                int_filter(1:self_coeff_list(i,1)) = 0;
                int_filters_list(i,1:N_integ_self,j) = int_filter;
            else
                int_filters_list(i,1:N_integ_self) = 0;
            end
        else
            if infer_couplings
                int_filter = coup_coeff_list(i,:,j)*basis_coupl;
                int_filters_list(i,1:N_integ_coupl,j) = int_filter;
            else
                int_filters_list(i,1:N_integ_coupl,j) = 0;
            end
        end
    end
end

filters = PACK_STRUCT('stim_filters_list', 'offset_list', ...
    'int_filters_list', 'stim_coeff_list', 'coup_coeff_list', ...
    'self_coeff_list');

end

%% Functions

function [loss, dloss] = llk(params, data_llk)
%%
UNPACK_STRUCT(data_llk)

% Negative log-likelihood
% =======================

% First we recover the interactions as a matrix to multiply to conv_mat_int
% to get the interaction & history potentials

N_basis = max(N_basis_self + 1, N_basis_coupl);
int_coeff = zeros(N_neur, N_basis);

if infer_couplings
    int_coeff = [];
    for i = 1:N_neur
        if i == icell
            if infer_self
                int_coeff(i, 1) = -1e3; % very large to approx. -Inf
                int_coeff(i, 2:N_basis_self+1) = params(1:N_basis_self);
                params(1:N_basis_self) = [];
            end
        else
            int_coeff(i, 1:N_basis_coupl) = params(1:N_basis_coupl);
            params(1:N_basis_coupl) = [];
        end
    end

    % Interaction and self potential
    h_int = squeeze(sum(int_coeff.*conv_mat_int, [1 2]));
    
elseif infer_self
    int_coeff = [-1e3 params(1:N_basis_self)];
    params(1:N_basis_self) = [];
    h_int = squeeze(sum(int_coeff.*conv_mat_int, [1 2]));
    
else
    h_int = 0;
    
end

% Stimulus parameters and stimulus potential
stim_coeff = reshape(params(1:Nx*Ny*N_basis_stim), N_basis_stim, Nx*Ny)';
h_stim = squeeze(sum(double(stim_coeff.*conv_mat_stim), [1 2]));

% Offset
h0 = params(end);

% Negative log-likelihood
exp_pot = exp(h0+h_stim+h_int);
loss = -(spikes_cell'*(h0+h_stim+h_int)-sum(exp_pot));

% Penalties
% =========

% Proxy parameters
lbd_proxy = 0.01;
lim_proxy = 10;

% L1 over couplings
if infer_couplings || infer_self
    proxy = zeros(size(int_coeff));
    proxy(abs(int_coeff./lbd_proxy)>lim_proxy) = abs(int_coeff(abs(int_coeff/lbd_proxy)>lim_proxy));
    proxy(abs(int_coeff./lbd_proxy)<=lim_proxy) = ...
        (log(1+exp(int_coeff(abs(int_coeff/lbd_proxy)<=lim_proxy)./lbd_proxy))...
        +log(1+exp(-int_coeff(abs(int_coeff/lbd_proxy)<=lim_proxy)./lbd_proxy)))*lbd_proxy;
    
    if ~infer_self
        proxy(icell,:) = 0;
    end    

    loss = loss + squeeze(sum((L1_int.*proxy), [1 2]));
end
    
% L1 over the stimulus filter
proxy = zeros(size(stim_coeff));
proxy(abs(stim_coeff./lbd_proxy)>lim_proxy) = abs(stim_coeff(abs(stim_coeff/lbd_proxy)>lim_proxy));
proxy(abs(stim_coeff./lbd_proxy)<=lim_proxy) = ...
    (log(1+exp(stim_coeff(abs(stim_coeff/lbd_proxy)<=lim_proxy)./lbd_proxy))...
    +log(1+exp(-stim_coeff(abs(stim_coeff/lbd_proxy)<=lim_proxy)./lbd_proxy)))*lbd_proxy;

loss = loss + squeeze(sum((L1_stim.*proxy), [1 2]));

% L2 over the spatial laplacian for the stimulus filter
if L2_lapl_stim > 0
    if Ny > 1
        lapl = lapl2D(reshape(stim_coeff, Nx, Ny, N_basis_stim), ...
            lapl_mat(:,:,1), lapl_mat(:,:,2));
        lapl2 = 2.*lapl2D(lapl, lapl_mat(:,:,1), lapl_mat(:,:,2));
        lapl2 = reshape(lapl2, Nx*Ny, N_basis_stim);
        loss = loss + L2_lapl_stim*sum(lapl.^2, [1 2 3]);
    elseif Ny == 1
        lapl = (stim_coeff'*lapl_mat)';
        lapl2 = 2.*(lapl'*lapl_mat)';
        loss = loss + L2_lapl_stim*sum(lapl.^2, [1 2]);       
    end
end

% Gradient of the log likelihood and penalties
% ============================================

temp = [];
temp(1,1,:) = spikes_cell - exp_pot;

if infer_couplings
    % Gradient regarding the couplings
    dloss_int = -sum(temp.*conv_mat_int, 3);
    % Gradient of the L1 penalty on the couplings
    dproxy = zeros(size(int_coeff));
    dproxy(abs(int_coeff)<lim_proxy) = tanh(int_coeff(abs(int_coeff)<lim_proxy)./(2*lbd_proxy));
    dproxy(abs(int_coeff)>=lim_proxy) = sign(int_coeff(abs(int_coeff)>=lim_proxy));
    
    if ~infer_self
        dproxy(icell,:) = 0;
        dloss_int(icell,:) = 0;
    end
    
    dloss_L1_int = L1_int.*dproxy;

elseif infer_self
    % Gradient regarding the couplings
    dloss_int = -sum(temp.*conv_mat_int, 3);
    % Gradient of the L1 penalty on the couplings
    dproxy = zeros(size(int_coeff));
    dproxy(abs(int_coeff)<lim_proxy) = tanh(int_coeff(abs(int_coeff)<lim_proxy)./(2*lbd_proxy));
    dproxy(abs(int_coeff)>=lim_proxy) = sign(int_coeff(abs(int_coeff)>=lim_proxy));
    
    dloss_L1_int = L1_int.*dproxy;
end

% Gradient regarding the stimulus filter
% temp is double and conv_mat_stim is single so some 
temp(temp>1e38) = 1e38;
temp(temp<-1e38) = -1e38;
dloss_stim = -sum(double(temp.*conv_mat_stim), 3);
dloss_stim(dloss_stim==Inf) = 1e38;
dloss_stim(dloss_stim==-Inf) = -1e38;

% Gradient of the L1 penalty on the stimulus filter
dproxy = zeros(size(stim_coeff));
dproxy(abs(stim_coeff)<lim_proxy) = tanh(stim_coeff(abs(stim_coeff)<lim_proxy)./(2*lbd_proxy));
dproxy(abs(stim_coeff)>=lim_proxy) = sign(stim_coeff(abs(stim_coeff)>=lim_proxy));
dloss_L1_stim = L1_stim*dproxy;
% Gradient of the L2 penalty on the stimulus spatial laplacian
dloss_L2_lapl_stim = zeros(size(stim_coeff));
if L2_lapl_stim > 0
    dloss_L2_lapl_stim = L2_lapl_stim*lapl2;
end
% Gradient of the offset
dloss_h0 = -sum(temp);

% Pack total total gradient
dloss = [];

% We add the interactions' gradients
if infer_couplings
    for i = 1:N_neur
        if i == icell
            if infer_self
                % Remember : the first coeff is not a parameter
                dloss = [dloss dloss_int(i,2:end)+dloss_L1_int(i,2:end)];
            end                
        else
            dloss = [dloss dloss_int(i,1:N_basis_coupl)+dloss_L1_int(i,1:N_basis_coupl)];
        end
    end
elseif infer_self
    dloss = [dloss dloss_int(1,2:end)+dloss_L1_int(1,2:end)];
end


% The sitmulus filters' gradients
dloss = [dloss reshape(dloss_stim'+dloss_L1_stim'+dloss_L2_lapl_stim', 1, N_basis_stim*Nx*Ny)];

% The offset gradient
dloss = [dloss dloss_h0];

end

% Matrix of the convolution of the spike trains by the couplings and spike
% history filter temporal bases
function conv_mat_int = conv_int(data_conv_int)

UNPACK_STRUCT(data_conv_int)

[N_neur, ~, N_repe] = size(spikes);

% Add basis element to account for the refractory period
basis_self = [[ones(1, refr_period) zeros(1, N_integ_self-refr_period)]' basis_self']';
basis_self(2:end, 1:refr_period) = 0;

N_basis = max(N_basis_coupl, N_basis_self + 1);
conv_mat_int = zeros(N_neur, N_basis, Nt_inf, N_repe);

for repe = 1:N_repe
    C = zeros(N_neur, N_basis, Nt_inf);
    % Couplings
    for i = 1:N_basis_coupl
        C(:,i,:) = conv2(squeeze(spikes(:,N_integ-N_integ_coupl+1:end-1,repe)), basis_coupl(i,:),'valid')*dt;
    end
    % self & refractory period
    cvtemp = conv2(1,squeeze(spikes(icell,N_integ-N_integ_self+1:end-1,repe)), basis_self)*dt;
    C(icell,:,:) = cvtemp(:,N_integ_self-1:end-N_integ_self);
    
    conv_mat_int(:,:,:,repe) = C;
end

conv_mat_int = reshape(conv_mat_int, N_neur, N_basis, Nt_inf*N_repe);
end

% Matrix of the convolution of the stimulus by the stimulus temporal basis
function conv_mat_stim = conv_stim(data_conv_stim)

UNPACK_STRUCT(data_conv_stim)

stimulus = single(reshape(stimulus,Nx*Ny,Nt_inf+N_integ_stim,N_repe)*2-1);

conv_mat_stim = zeros(Nx*Ny, Nt_inf, N_basis_stim, N_repe, 'single');
for repe = 1:N_repe
    for i = 1:N_basis_stim
        cvtemp = single(conv2(1, basis_stim(i,:), squeeze(stimulus(:,:,repe)), 'valid')*dt);
        conv_mat_stim(:,:,i,repe) = cvtemp(:,1:end-1);
    end
end
conv_mat_stim = permute(conv_mat_stim, [1 3 2 4]);
conv_mat_stim = reshape(conv_mat_stim, Nx*Ny, N_basis_stim, Nt_inf*N_repe);

end

% Computes the laplacian of a NxxNyxNt array with periodic border conditions
% Input matrix has the shape Nx Ny Nt
function lapl = lapl2D(M, lapl_mat_x, lapl_mat_y)
    [Nx, Ny, N] = size(M);
    M_temp = reshape(permute(M, [2 1 3]), Ny*Nx, N);
    M = reshape(M, Nx*Ny, N);
    lapl_x = (lapl_mat_x*M);
    lapl_y = (lapl_mat_y*M_temp);
    lapl = reshape(lapl_x, Nx,Ny,N) +  ...
        permute(reshape(lapl_y, Nx,Ny,N),[2 1 3]);
end