function [filters, stim_potential] = interactions_fit(data_fit)
%% Unpack data
UNPACK_STRUCT(data_fit)

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
    spikes_cell = squeeze(spikes(icell,N_integ+1:end,:));

    % Initial guess for the parameters
    stim_potential = stim_potential_list(icell,:);
    coup_coeff = squeeze(coup_coeff_list(icell,:,:));
    self_coeff = squeeze(self_coeff_list(icell,:));

    % Packing the parameters in vector
    params = [];
    % Couplings and Spike History Filter coefficients
    for i = 1:N_neur
        if i == icell
            % ! First param is the fixed refractory period
            params = [params self_coeff(2:end)];
            refr_period = self_coeff(1);
        else
            params = [params coup_coeff(:,i)'];
        end
    end
    % Stimulus potential
    params = [params stim_potential];

    % Useful matrix for the computation of the interaction effects
    data_conv_int = PACK_STRUCT('icell', 'spikes', 'dt', 'Nt_inf', ...
        'basis_coupl', 'N_basis_coupl', 'N_integ_coupl', ...
        'basis_self', 'N_basis_self', 'N_integ_self', ...
        'refr_period', 'N_integ');
    conv_mat_int = conv_int(data_conv_int);
    
    % Minimization of the log-likelihood
    data_llk = PACK_STRUCT('icell', 'spikes_cell', 'N_neur', 'dt', ...
        'N_basis_coupl', 'N_basis_self', 'conv_mat_int', ...
        'L1_int', 'L2_stim_pot', 'L2_lapl_stim_pot');
    params = fminunc(@(x)llk(x, data_llk), params, opts);

    % Recover the parameters
    for i = 1:N_neur
        if i == icell
            % ! First coeff is the refractory period
            self_coeff_list(icell,2:end) = params(1:N_basis_self);
            params(1:N_basis_self) = [];
        else
            coup_coeff_list(icell,:,i) = params(1:N_basis_coupl);
            params(1:N_basis_coupl) = [];
        end
    end
    stim_potential_list(icell,:) = params;           
end

%% We reconstruct the filters

% Interactions and self
int_filters_list = zeros(N_neur,max(N_integ_coupl, N_integ_self), N_neur);
for i = 1:N_neur
    for j = 1:N_neur
        % filter from j to i
        if j == i
            int_filter = self_coeff_list(i,2:end)*basis_self;
            int_filter(1:self_coeff_list(i,1)) = 0;
            int_filters_list(i,1:N_integ_self,j) = int_filter;
        else
            int_filter = coup_coeff_list(i,:,j)*basis_coupl;
            int_filters_list(i,1:N_integ_coupl,j) = int_filter;
        end
    end
end

filters = PACK_STRUCT('stim_potential_list', 'int_filters_list', ...
    'coup_coeff_list', 'self_coeff_list');

end

%% Functions

function [loss, dloss] = llk(params, data_llk)

UNPACK_STRUCT(data_llk)

% Negative log-likelihood
% =======================

% Recover the interactions as a matrix, multiply it by conv_mat_int
% to get the couplings & spike history effects

N_basis = max(N_basis_self + 1, N_basis_coupl);
int_coeff = zeros(N_neur, N_basis);
for i = 1:N_neur
    if i == icell
        int_coeff(i, 1) = -1e3; % very large to approx. -Inf
        int_coeff(i, 2:N_basis_self+1) = params(1:N_basis_self);
        params(1:N_basis_self) = [];
    else
        int_coeff(i, 1:N_basis_coupl) = params(1:N_basis_coupl);
        params(1:N_basis_coupl) = [];
    end
end

% Interaction and self potential
h_int = squeeze(sum(int_coeff.*conv_mat_int, [1 2]));

% Stimulus potential
h_stim = params';

% Negative log-likelihood
exp_pot = exp(h_stim + h_int);

loss = -sum(spikes_cell.*(h_stim + h_int) - exp_pot, [1 2]);

% Penalties
% =========

% L1 approx. parameters
lbd_approx = 0.01;
lim_approx = 10;

% Approx. L1 over couplings
proxy = zeros(size(int_coeff));
proxy(abs(int_coeff./lbd_approx)>lim_approx) = abs(int_coeff(abs(int_coeff/lbd_approx)>lim_approx));
proxy(abs(int_coeff./lbd_approx)<=lim_approx) = ...
    (log(1+exp(int_coeff(abs(int_coeff/lbd_approx)<=lim_approx)./lbd_approx))...
    +log(1+exp(-int_coeff(abs(int_coeff/lbd_approx)<=lim_approx)./lbd_approx)))*lbd_approx;
proxy(icell, 1) = 0;
loss = loss + squeeze(sum((L1_int.*proxy), [1 2]));

% L2 over the stimulus potential
loss = loss + squeeze(sum((L2_stim_pot.*h_stim).^2));

% L2 of the laplacian of the stimulus potential
loss = loss + squeeze(sum(L2_lapl_stim_pot.*(4.*del2(h_stim)).^2));

% Gradient of the log likelihood and penalties
% ============================================

temp = [];
temp(1,1,:,:) = spikes_cell - exp_pot;

% Gradient of the couplings
dloss_int = -sum(temp.*conv_mat_int, [3 4]);
dloss_int(icell,1) = 0;
% Gradient of the L1 penalty on the couplings
dproxy = zeros(size(int_coeff));
dproxy(abs(int_coeff)<lim_approx) = tanh(int_coeff(abs(int_coeff)<lim_approx)./(2*lbd_approx));
dproxy(abs(int_coeff)>=lim_approx) = sign(int_coeff(abs(int_coeff)>=lim_approx));
dloss_L1_int = L1_int.*dproxy;
dloss_L1_int(icell,1) = 0;
% Gradient of the stimulus potential 
dloss_stim = -sum(squeeze(temp), 2)';
% Gradient of the L2 penalty on the stimulus potential
dloss_L2_stim = 2*L2_stim_pot*h_stim';
% Gradient of the L2 laplacian on the stimulus potential
dloss_L2_lapl_stim = 2*L2_lapl_stim_pot*(4*del2(4*del2(h_stim)));

% Pack total total gradient
dloss = [];

% We add the interactions' gradients
for i = 1:N_neur
    if i == icell
        % Remember : the first coeff is not a parameter
        dloss = [dloss dloss_int(i,2:N_basis_self+1)+dloss_L1_int(i,2:N_basis_self+1)];
    else
        dloss = [dloss dloss_int(i,1:N_basis_coupl)+dloss_L1_int(i,1:N_basis_coupl)];
    end
end

% We add the stimulus potential's gradient
dloss = [dloss dloss_stim + dloss_L2_stim + dloss_L2_lapl_stim'];

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

end