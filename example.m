clear all
close all

addpath ./helpers

%% Loading the models

infer_models = false; % reinfer the models if true, load the pretrained models if false
% Note that reinfering all the models requires several Gbs of free ram and can take several hours.

if infer_models == true
    run inference.m
    % Two-step model
    int_model = load('./infered_models/infered_interactions_bars.mat').infered_interactions;
    LN_model = load('./infered_models/infered_LN_bars.mat').infered_model;
    % Classicaly infered GLM model
    GLM_model = load('./infered_models/infered_GLM_bars.mat').infered_model;    
else
    % Two-step model
    int_model = load('./infered_models/example/infered_interactions_bars_ex.mat').infered_interactions;
    LN_model = load('./infered_models/example/infered_LN_bars_ex.mat').infered_model;
    % Classicaly infered GLM model
    GLM_model = load('./infered_models/example/infered_GLM_bars_ex.mat').infered_model;
end

%% Load testing set

load('./data/test_data_bars.mat')

%% Simulating the classical GLM

% Some parameters
cells = GLM_model.parameters.cells;
dt = GLM_model.parameters.dt;
N_neur = numel(cells);
N_integ_stim = GLM_model.parameters.N_integ_stim;
N_test = size(spikes_test, 2);
Nt_sim = N_test - N_integ_stim;
N_integ = max(GLM_model.parameters.N_integ_coupl, GLM_model.parameters.N_integ_self);

% Compute the stimulus potential
stim_filters_list = GLM_model.filters.stim_filters_list;
stim_pot_data = PACK_STRUCT('stimulus_test', 'stim_filters_list', 'cells', 'dt');
h_stim = stimulus_potential(stim_pot_data);
offset_list = GLM_model.filters.offset_list;

% Interaction model
int_filters = GLM_model.filters.int_filters_list;
tau_r = GLM_model.filters.self_coeff_list(:,1);

% Simulation
N_repe_test = size(spikes_test, 3);
spikes_glm = zeros(N_neur, Nt_sim, N_repe_test);
h_list = zeros(N_neur, Nt_sim, N_repe_test);

for repe = 1:N_repe_test
    repe
    spikes_temp = zeros(N_neur, Nt_sim + N_integ);
    % Init with the actual data for the spike histories before simulation
    spikes_temp(:,1:N_integ) = spikes_test(:, N_integ_stim-N_integ+1:N_integ_stim, 1);
    for t = 1:Nt_sim
        % Interaction effects and mean-field correction
        h_int = interation_potential(t, spikes_temp, dt, int_filters, N_integ);
        % Predicted log(psth) and firing rate
        h_list(:,t,repe) = offset_list + h_stim(:,t) + h_int;
        lbd = exp(h_list(:,t,repe));
        % Hard refractory period
        for icell = 1:N_neur
            if sum(spikes_temp(icell,N_integ+t-tau_r(icell):N_integ+t-1))>0
                lbd(icell) = 0;
            end
        end
        % Sort the spikes
        spikes_temp(:,N_integ+t) = rand(N_neur, 1)<lbd;
    end
    spikes_glm(:,:,repe) = spikes_temp(:,N_integ+1:end);
end


%% Simulating the 2-steps GLM

% Some parameters
cells = LN_model.parameters.cells;
dt = LN_model.parameters.dt;
N_neur = numel(cells);
N_integ_stim = LN_model.parameters.N_integ_stim;
N_test = size(spikes_test, 2);
Nt_sim = N_test - N_integ_stim;
N_integ = max(int_model.parameters.basis.N_integ_coupl, int_model.parameters.basis.N_integ_self);

% Compute the stimulus potential
stim_filters_list = LN_model.filters.stim_filters_list;
stim_pot_data = PACK_STRUCT('stimulus_test', 'stim_filters_list', 'cells', 'dt');
h_stim = stimulus_potential(stim_pot_data);
offset_list = LN_model.filters.offset_list;

% Interaction model
int_filters = int_model.filters.int_filters_list;
tau_r = int_model.filters.self_coeff_list(:,1);

% Correction for the effect of the refractory period
pred_pot = exp(offset_list + h_stim);
corr_fact = zeros(size(pred_pot));
for i = 1:N_neur
    p_temp = [zeros(1, tau_r(i)) pred_pot(i,:)];
    for t = 1:size(pred_pot,2)
        corr_fact(i,t) = sum(p_temp(t:t+tau_r(i)-1));
    end
end

% Simulation
N_repe_test = size(spikes_test, 3);
spikes_2steps = zeros(N_neur, Nt_sim, N_repe_test);
h_list = zeros(N_neur, Nt_sim, N_repe_test);

for repe = 1:N_repe_test
    repe
    spikes_temp = zeros(N_neur, Nt_sim + N_integ);
    % Init with the actual data for the spike histories before simulation
    spikes_temp(:,1:N_integ) = spikes_test(:, N_integ_stim-N_integ+1:N_integ_stim, 1);
    % Init potential for the GLM to be corrected
    h_pred = log(mean(spikes_test(:,N_integ_stim-N_integ+1:N_integ_stim,:), 3)+1e-6);
    
    for t = 1:Nt_sim
        % Predicted log(psth) by the LN model (or by any other model !)
        h_pred = [h_pred offset_list+h_stim(:,t)];
        % Interaction effects and mean-field correction to h_pred
        h_int = interation_potential_2steps(t, spikes_temp, dt, ...
            int_filters, N_integ, h_pred, corr_fact);
        % Predicted log(psth) and firing rate
        h_list(:,t,repe) = offset_list + h_stim(:,t) + h_int;
        lbd = exp(h_list(:,t,repe));
        % Hard refractory period
        for icell = 1:N_neur
            if sum(spikes_temp(icell,N_integ+t-tau_r(icell):N_integ+t-1))>0
                lbd(icell) = 0;
            end
        end
        % Sort the spikes
        spikes_temp(:,N_integ+t) = rand(N_neur, 1)<lbd;
    end
    spikes_2steps(:,:,repe) = spikes_temp(:,N_integ+1:end);
end

%% Computing the PSTH and correlations

% PSTH
tlist = [1:Nt_sim]*dt/1e3;
psth_data = squeeze(mean(spikes_test(:,N_integ_stim+1:end,:),3));
psth_glm = squeeze(mean(spikes_glm, 3));
psth_2steps = squeeze(mean(spikes_2steps, 3));

for neur = 1:N_neur
    psth_data(neur,:) = smooth(psth_data(neur,:), 20)*1e3/dt;
    psth_glm(neur,:) = smooth(psth_glm(neur,:), 20)*1e3/dt;
    psth_2steps(neur,:) = smooth(psth_2steps(neur,:), 20)*1e3/dt;
end

% Correlations
load('./data/repeat_data_bars.mat', 'Xell', 'Yell')
bin_size = 25; % bin size for the computations of the correlations
max_corr_time = 10; % max delay to which compute the correlations
cov_bool = false; % if true computes covariance instead of correlations

% Criterion to discard simulations with unrealistic activity
max_spikes = round(mean(sum(spikes_test, 2), [1 3]) + 3*std(sum(spikes_test, 2), [], [1 3]));

disp('Compute noise corr : data')
corr_data = compute_corr(spikes_test(:,N_integ_stim+1:end,:), bin_size, max_corr_time, Xell, Yell, cov_bool, max_spikes);
disp('Compute noise corr : max-L GLM')
corr_glm = compute_corr(spikes_glm, bin_size, max_corr_time, Xell, Yell, cov_bool, max_spikes);
disp('Compute noise corr : 2steps GLM')
corr_2steps = compute_corr(spikes_2steps, bin_size, max_corr_time, Xell, Yell, cov_bool, max_spikes);

%% Some plots

% Firing rates
figure
for i = 1:N_neur
subplot(2,3,i)
hold on
plot(psth_data(i,:), 'LineWidth', 2)
plot(psth_glm(i,:), 'LineWidth', 2)
plot(psth_2steps(i,:), 'LineWidth', 2)
if i == 1; legend(["data", "log-$\ell$", "2-steps"], 'Interpreter', 'latex'); end
xlabel('time (s)')
ylabel('firing rate (Hz)')
xlim([0 4000])
end

% Noise correlations
figure
hold on
plot(corr_data.noise_corr, corr_2steps.noise_corr, '.', 'MarkerSize', 15)
plot(corr_data.noise_corr, corr_glm.noise_corr, '.', 'MarkerSize', 15)
plot([-0.02 0.22], [-0.02 0.22], '--k', 'LineWidth', 3)
xlabel('data')
ylabel('prediction')
legend(["2-steps", "log-$\ell$"], 'Interpreter', 'latex')
xlim([-0.02 0.22])

%% Empt stim corr VS noise corr

figure;
hold on
plot( corr_data.stim_corr, ( (corr_2steps.noise_corr - corr_data.noise_corr)./max(3*std(corr_data.noise_corr),corr_data.noise_corr) ), '.', 'MarkerSize', 15)
plot( corr_data.stim_corr, ( (corr_glm.noise_corr - corr_data.noise_corr)./max(3*std(corr_data.noise_corr),corr_data.noise_corr) ), '.', 'MarkerSize', 15)
plot([-0.05 0.7], [0 0], '--k', 'LineWidth', 3)
ylim([-1 4])
xlim([-0.05 0.7])
xlabel('Emp. Stim. Corr.')
ylabel('Norm. Rel. Error')

%% Population activity

figure
subplot(1,3,1)
imagesc(squeeze(mean(spikes_test(:,N_integ_stim+1:end,:),1))')
title('data')
subplot(1,3,2)
imagesc(squeeze(mean(spikes_glm,1))')
title('max log-l')
subplot(1,3,3)
imagesc(squeeze(mean(spikes_2steps,1))')
title('two-step')


%% Helper functions

% Stimulus potential
function h = stimulus_potential(stim_pot_data)
UNPACK_STRUCT(stim_pot_data)

stimulus = double(stimulus_test)*2-1;
h = zeros(size(stim_filters_list, 4), size(stimulus, 3) - size(stim_filters_list, 3));
for cell = cells
    for x = 1:size(stimulus,1)
        for y = 1:size(stimulus,2)
            cvtemp = conv(squeeze(stimulus(x,y,:)), squeeze(stim_filters_list(x,y,:,cell)), 'valid');
            h(cell,:) = h(cell,:) + cvtemp(1:end-1)'.*dt;
        end
    end    
end
end

% Interaction potential
function h = interation_potential_2steps(t, spikes_temp, dt, int_filters, N_integ, ...
    h_psth, corr_fact)

t = t + N_integ;

temp1 = [];
temp2 = [];

temp1(1,:,:) = spikes_temp(:,t-N_integ:t-1)';
temp2(1,:,:) = exp(h_psth(:,t-N_integ:t-1))';

h = sum(int_filters(:,end:-1:1,:).*(temp1-temp2), [2 3]).*dt;
h = h + corr_fact(:,t-N_integ);

end

function h = interation_potential(t, spikes_temp, dt, int_filters, N_integ)

t = t + N_integ;

temp = [];
temp(1,:,:) = spikes_temp(:,t-N_integ:t-1)';

h = sum(int_filters(:,end:-1:1,:).*temp, [2 3]).*dt;

end
