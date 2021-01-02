
addpath ./helpers

%% Parameters and regularization of the inference

% General
dt = 1.667; %ms

% Stimulus filter dimensions
N_integ_stim = 200;
% Stimulus filter RC basis
first_peak_stim = 1;
last_peak_stim = 170.;
streach_stim = 50;
N_basis_stim = 6;

% Coupling filters
% Filter dimension
N_integ_coupl = 25;
% Couplings RC basis
first_peak_coupl = 0;
last_peak_coupl = 15.;
streach_coupl = 5.;
N_basis_coupl = 4;

% Post spike filter
% Filter dimension
N_integ_self = 25;
first_peak_self = 1;
last_peak_self = 15;
streach_self = 1.;
N_basis_self = 7;

% Regularization
% Interactions
L1_coupl = 0.1; % on the couplings
L1_self = 0.; % on the self coupling

% Two-steps auxiliary stimulus parameter
L2_stim_pot = 1e-6;
L2_lapl_stim_pot = 0.;

% Stimulus filters
L1_stim = 50.;
L2_lapl_stim = 500.; % L2 on the spatial laplacian of the stimulus filter

N_integ = max(N_integ_coupl, N_integ_self);

%% Generate the bases functions for the couplings and the stimulus filters

% Spikes train to estimate the refractory period
load('./data/nonrepeat_data_bars.mat', 'spikes_train')

data_couplings_basis = PACK_STRUCT('first_peak_coupl', 'last_peak_coupl', ...
    'streach_coupl', 'N_basis_coupl', 'N_integ_coupl', ...
    'first_peak_self', 'last_peak_self', ...
    'streach_self', 'N_basis_self', 'N_integ_self', ...
    'spikes_train');

UNPACK_STRUCT(gen_basis_couplings(data_couplings_basis));

data_stimulus_basis = PACK_STRUCT('first_peak_stim', 'last_peak_stim', ...
    'streach_stim', 'N_basis_stim', 'N_integ_stim');

UNPACK_STRUCT(gen_basis_stim(data_stimulus_basis));

%% Load data for the 2-step inference

load('./data/repeat_data_bars.mat')
spikes = spikes_train; [N_neur, Nt, N_repe] = size(spikes); Nt_inf = Nt - N_integ;
cells = 1:N_neur; % Cells to fit

%% Inference couplings for the 2-steps procedure

% Filters initial guess

% Start anew
stim_potential_list = log(mean(spikes(:,N_integ+1:end,:), 3)+1e-6);
coup_coeff_list = zeros(N_neur, N_basis_coupl, N_neur);
self_coeff_list = zeros(N_neur, N_basis_self+1);
self_coeff_list(:,1) = tau_r; % First self coeff is the refractory period

% Load results of a previous inference
% load('./infered_models/example/infered_interactions_bars_ex.mat')
% stim_potential_list = infered_interactions.filters.stim_potential_list;
% coup_coeff_list = infered_interactions.filters.coup_coeff_list;
% self_coeff_list = infered_interactions.filters.self_coeff_list;

% Inference

data_fit = PACK_STRUCT('L1_coupl', 'L1_self', 'dt', ...
        'N_neur', 'basis_self', 'basis_coupl', 'N_repe', 'L2_stim_pot', ...
        'Nt_inf', 'cells', 'spikes', 'N_basis_coupl', 'L2_lapl_stim_pot', ...
        'N_integ', 'stim_potential_list', 'coup_coeff_list', ...
        'self_coeff_list', 'N_basis_self', 'N_integ_self', 'N_integ_coupl');

filters = interactions_fit(data_fit);

% Store the data

inf_date = date;

parameters = struct();
parameters.global = PACK_STRUCT('Nt_inf', 'cells', 'dt', 'N_repe');
parameters.basis = PACK_STRUCT('N_integ_coupl', 'N_basis_coupl', ...
    'first_peak_coupl', 'last_peak_coupl', 'streach_coupl', ...
    'N_integ_self', 'N_basis_self', 'first_peak_self', ...
    'last_peak_self', 'streach_self', 'tau_r');
parameters.regu = PACK_STRUCT('L1_coupl', 'L1_self', 'L2_stim_pot', 'L2_lapl_stim_pot');

infered_interactions = PACK_STRUCT('filters', 'parameters', 'inf_date');

save('./infered_models/infered_interactions_bars.mat', 'infered_interactions')

%% Load data for the LN and GLM inference

load('./data/nonrepeat_data_bars.mat')
spikes = spikes_train; [N_neur, Nt, N_repe] = size(spikes); Nt_inf = Nt - N_integ_stim;
cells = 1:N_neur; % Cells to fit
spikes = spikes(:,N_integ_stim-N_integ+1:end,:);
[Nx, Ny, ~, ~] = size(stimulus);

%% Infer LN model

% Infer no couplings and no self coupling
infer_couplings = false;
infer_self = false;

% Filters initial guess

% Start anew
stim_coeff_list = zeros(N_neur, Nx*Ny, N_basis_stim);
offset_list = log(mean(spikes_train, [2 3]));
coup_coeff_list = zeros(N_neur, N_basis_coupl, N_neur);
self_coeff_list = zeros(N_neur, N_basis_self+1);
self_coeff_list(:,1) = tau_r; % First self coeff is the refractory period

% Load results of a previous inference
% load('./infered_models/example/infered_LN_bars_ex.mat')
% stim_coeff_list = infered_model.filters.stim_coeff_list;
% offset_list = infered_model.filters.offset_list;
% coup_coeff_list = infered_model.filters.coup_coeff_list;
% self_coeff_list = infered_model.filters.self_coeff_list;

% Inference

data_fit = PACK_STRUCT('spikes', 'stimulus', 'cells', 'Nt_inf', 'dt', ...
        'offset_list', 'stim_coeff_list', 'coup_coeff_list', 'self_coeff_list', ...
        'basis_stim', 'N_basis_stim', 'N_integ_stim', ...
        'infer_couplings', 'infer_self', 'N_integ', ...
        'basis_coupl', 'N_basis_coupl', 'N_integ_coupl', ...
        'basis_self', 'N_basis_self', 'N_integ_self', ...
        'L1_coupl', 'L1_self', 'L1_stim', 'L2_lapl_stim');

filters = glm_fit(data_fit);

% Store data

inf_date = date;
parameters = PACK_STRUCT('cells', 'Nt_inf', 'dt', 'N_repe', 'infer_couplings', 'infer_self', ...
    'Nx', 'Ny', 'first_peak_stim', 'last_peak_stim', 'streach_stim', 'N_integ_stim', 'N_basis_stim', ...
    'first_peak_coupl', 'last_peak_coupl', 'streach_coupl', 'N_integ_coupl', 'N_basis_coupl', ...
    'first_peak_self', 'last_peak_self', 'streach_coupl', 'N_integ_self', 'N_basis_self', ...
    'L1_coupl', 'L1_self', 'L1_stim', 'L2_lapl_stim');

infered_model = PACK_STRUCT('filters', 'parameters', 'inf_date');

save('./infered_models/infered_LN_bars.mat', 'infered_model')

%% Infer GLM

% Infer couplings and self coupling ?
infer_couplings = true;
infer_self = true;

% Filters initial guess

% Start anew
% stim_coeff_list = zeros(N_neur, Nx*Ny, N_basis_stim);
% offset_list = log(mean(spikes_train, [2 3]));
% coup_coeff_list = zeros(N_neur, N_basis_coupl, N_neur);
% self_coeff_list = zeros(N_neur, N_basis_self+1);
% self_coeff_list(:,1) = tau_r; % First self coeff is the refractory period

% Use the LN stimulus filter and the 2-steps inference couplings as a starting point
load('./infered_models/infered_LN_bars.mat')
stim_coeff_list = infered_model.filters.stim_coeff_list;
offset_list = infered_model.filters.offset_list;
load('./infered_models/infered_interactions_bars.mat')
coup_coeff_list = infered_interactions.filters.coup_coeff_list;
self_coeff_list = infered_interactions.filters.self_coeff_list;

% Use a previously infered model as a starting point
% load('./infered_models/example/infered_GLM_bars_ex.mat')
% stim_coeff_list = infered_model.filters.stim_coeff_list;
% offset_list = infered_model.filters.offset_list;
% coup_coeff_list = infered_model.filters.coup_coeff_list;
% self_coeff_list = infered_model.filters.self_coeff_list;

% Inference

data_fit = PACK_STRUCT('spikes', 'stimulus', 'cells', 'Nt_inf', 'dt', ...
        'offset_list', 'stim_coeff_list', 'coup_coeff_list', 'self_coeff_list', ...
        'basis_stim', 'N_basis_stim', 'N_integ_stim', ...
        'infer_couplings', 'infer_self', 'N_integ', ...
        'basis_coupl', 'N_basis_coupl', 'N_integ_coupl', ...
        'basis_self', 'N_basis_self', 'N_integ_self', ...
        'L1_coupl', 'L1_self', 'L1_stim', 'L2_lapl_stim');

filters = glm_fit(data_fit);

% Store data

inf_date = date;
parameters = PACK_STRUCT('cells', 'Nt_inf', 'dt', 'N_repe', 'infer_couplings', 'infer_self', ...
    'Nx', 'Ny', 'first_peak_stim', 'last_peak_stim', 'streach_stim', 'N_integ_stim', 'N_basis_stim', ...
    'first_peak_coupl', 'last_peak_coupl', 'streach_coupl', 'N_integ_coupl', 'N_basis_coupl', ...
    'first_peak_self', 'last_peak_self', 'streach_coupl', 'N_integ_self', 'N_basis_self', ...
    'L1_coupl', 'L1_self', 'L1_stim', 'L2_lapl_stim');

infered_model = PACK_STRUCT('filters', 'parameters', 'inf_date');

save('./infered_models/infered_GLM_bars.mat', 'infered_model')