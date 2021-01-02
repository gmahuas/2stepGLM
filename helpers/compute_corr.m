function correlations = compute_corr(spikes, bin_size, max_corr_time, Xell, Yell, cov_bool, max_spikes)

[N_neur, Nt, N_repe] = size(spikes);

%% Work on the good repetitions only

good_rep_list = {};

for neur = 1:N_neur
    sat_rep = [];
    for repe = 1:N_repe
        if sum(spikes(neur,:,repe))>max_spikes
            sat_rep = [sat_rep repe];
            sum(spikes(neur,:,repe));
        end
    end
    good_rep = setdiff([1:N_repe], sat_rep);
    good_rep_list{neur} = good_rep;
end

sat_list = [];
for i = 1:N_neur
    sat_list = [sat_list setdiff([1:N_repe], good_rep_list{i})];    
end
sat_list = unique(sat_list);
good_list = setdiff([1:N_repe], sat_list);

if ~isempty(sat_list)
    disp(['Repetition(s) ' num2str(sat_list) ' were discarded, had more than ' num2str(max_spikes) ' spikes']) 
end

N_repe = numel(good_list);
spikes = spikes(:,:,good_list);

%% We bin the spike response and build the spike count matrix

N_bins_psth = floor(Nt/bin_size);
tronc = mod(Nt, bin_size);

% In order to have N_bins_psth pair (so that we can easily get the center
% bin corresponding to the zero time lag noise covariance
if mod(N_bins_psth, 2) == 1
    N_bins_psth = N_bins_psth - 1;
    tronc = tronc + bin_size;
end

spikes = reshape(full(spikes), N_neur, Nt, N_repe);
spikes = reshape(spikes(:,1:end-tronc,:), N_neur, (Nt-tronc)*N_repe);

n = squeeze(sum(reshape(spikes, N_neur, bin_size, N_bins_psth*N_repe), 2, 'native'));
n = reshape(n, N_neur, N_bins_psth, N_repe);
n = double(n);

%% Covariances

% Mean spike count across repetitions
lambda = mean(n, 3);
% Mean spike count for each neuron
mean_lambda = mean(lambda, [2]);

% Stimulus covariance
z_lambda = lambda - repmat(mean_lambda, 1, N_bins_psth);
C_stim = zeros(N_neur, N_neur, max_corr_time+1);
for tau = -max_corr_time:max_corr_time
    C_stim(:,:,max_corr_time+1+tau) = (1/N_bins_psth).*(z_lambda*circshift(z_lambda, tau, 2)');
end

% Total covariance
z_n = n - repmat(mean_lambda, 1, N_bins_psth, N_repe);
C_tot = zeros(N_neur, N_neur, max_corr_time+1);
C_tot_temp = zeros(N_neur, N_neur, N_repe);
for tau = -max_corr_time:max_corr_time
    for repe = 1:N_repe
        C_tot_temp(:,:,repe) = (1/N_bins_psth).*(z_n(:,:,repe)*...
            circshift(z_n(:,:,repe), tau, 2)');
    end 
    C_tot(:,:,max_corr_time+1+tau) = mean(C_tot_temp, 3);
end

% Noise covariance
n_l = n - lambda;
C_noise = zeros(N_neur, N_neur, max_corr_time+1);
nlnl = zeros(N_neur, N_neur, N_repe);

for tau = -max_corr_time:max_corr_time
    n_l_shift = circshift(n_l, tau, 2);
    for repe = 1:N_repe
        nlnl(:,:,repe) = n_l(:,:,repe)*n_l_shift(:,:,repe)'/N_bins_psth;
    end
    C_noise(:,:,max_corr_time+1+tau) = sum(nlnl, 3)/N_repe;
end

%% Correlations with the distance

% First we have to compute the positions of the cells

Xell_ = (Xell-15.5)*20*3.3;
Yell_ = (Yell-15.5)*20*3.3;

Xcenter = mean( Xell_ );
Ycenter = mean( Yell_ );

% Correlations with the distance

stim_corr = [];
noise_corr = [];

dist_cells_x = [];
dist_cells_y = [];
dist_cells = [];

if ~cov_bool
    for i = 2:N_neur
        for j = 1:i-1
            noise_corr(end+1) = C_noise(i,j,max_corr_time+1)/sqrt(C_tot(i,i,max_corr_time+1)*C_tot(j,j,max_corr_time+1));
            stim_corr(end+1) = C_stim(i,j,max_corr_time+1)/sqrt(C_tot(i,i,max_corr_time+1)*C_tot(j,j,max_corr_time+1));
            dist_cells_x(end+1) = sqrt((Xcenter(i)-Xcenter(j))^2);
            dist_cells_y(end+1) = sqrt((Ycenter(i)-Ycenter(j))^2);
            dist_cells(end+1) = sqrt((Xcenter(i)-Xcenter(j))^2+(Ycenter(i)-Ycenter(j))^2);
        end
    end
elseif cov_bool
    for i = 2:N_neur
        for j = 1:i-1
            noise_corr(end+1) = C_noise(i,j,max_corr_time+1);
            stim_corr(end+1) = C_stim(i,j,max_corr_time+1);
            dist_cells_x(end+1) = sqrt((Xcenter(i)-Xcenter(j))^2);
            dist_cells_y(end+1) = sqrt((Ycenter(i)-Ycenter(j))^2);
            dist_cells(end+1) = sqrt((Xcenter(i)-Xcenter(j))^2+(Ycenter(i)-Ycenter(j))^2);
        end
    end
end


%% Store the data

tot_corr = stim_corr+noise_corr;

correlations = PACK_STRUCT('C_noise', 'C_stim', 'C_tot', ...
    'noise_corr', 'stim_corr', 'tot_corr', ...
    'dist_cells_x', 'dist_cells_y', 'dist_cells');

disp('Done')

end
