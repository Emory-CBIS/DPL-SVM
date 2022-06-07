function [misc_err_rate,beta_posterior_mean,samp1,beta_store] = DPLSVM(Omega,scov,Ylabel)

%% data input 
% N : total number of subjects
% Q : total number of screened edges per subject
% C : total number of covariates per subject
% P : total number of varialbes (number of screened edges + number of covariates)
% Omega : N * Q matrix of network information
% scov  : N * C matric of covariates 
% Yi    : N * 1 matrix of label
N0 = size(Omega,1);
V1 = size(Omega,2);
P0 = size(scov,2);
P = V1+P0;
Xall = zeros(N0,P);
Xall(:,1:V1) = Omega;
Xall(:,(V1+1):P) = scov;
samp0 = 1:N0;
samp1 = sort(randsample(N0,N));
samp2 = setdiff(samp0,samp1);
X = Xall(samp0,:);
Xtest = Xall(samp1,:);


Yi = Ylabel(samp1);
Ytest = Ylabel(samp2);

P = size(X,2);

%% Testing DPM with Laplace base measure in standard linear regression
% context

%% Edit settings below
%N             = 100; % Number of subjects
%P             = 200; % Number of covariate effects
%true_sigma_sq = 0.8; % Residual error variance

nBurnin = 3000;
nMCMC   = 3000;

n_start_cluster = 5; % number of clusters at start of posterior computation. NOT true number of clusters
Conc = 1;          % concentration parameter

% base measure shape and rate - hyperparameters
r_lambda = 1.1;
delta_lambda = 10.0;

%% Data Generation - User Settings
% Two main setups
% Setup 1: Specify lambda (squared) values. Data are generated following
% the assumed model. This works fine, but tends to still result in
% relatively small beta values, and often the clustering does not seem to
% work that well
%
% Setup 2: Specify means for the beta terms - this can work a little better
% for making sure some of the betas are actually "large". Basically defines
% clusters of betas with different mu and sd

% Define setup
setup = 1;

% Edit these for setup 1
lambda_sq_true = [1, 1000];

% Edit these for setup 2, all should have the same number of elements
%beta_mu   = [0.0; 3.0; 60.0]; % mean for each cluster.
%beta_sd   = [0.000001; 0.1; 0.1];
%prop_memb = [0.9; 0.05; 0.05]; % prop data in each cluster

%% You should not need to edit anything below this point

%% Data Generation based on provided values (should not need to edit)

% Generate covariate values
%X = rand(N, P);
%for p = 1:P; X(:, p) = (X(:, p) - mean(X(:, p))) / std(X(:, p)); end

% Generate beta values - the way these are generated depends on
% user-defined setup
%if setup == 1
 %   n_true_cluster = length(lambda_sq_true);
  %  true_cluster_membership = datasample(1:n_true_cluster, P);
  %  sigma_sq_beta_true = zeros(P, 1);
  %  beta_true = zeros(P, 1);
  %  for p = 1:P
        % Generate corresponding beta values
   %     sigma_sq_beta_true(p) = gamrnd(1, 2/lambda_sq_true(true_cluster_membership(p)));
   %     beta_true(p) = normrnd(0.0, sqrt(sigma_sq_beta_true(p))) ;
   % end
%else
%    n_true_cluster = length(beta_mu);
%    prop_memb = prop_memb / sum(prop_memb); % in case mental math mistake
%    true_cluster_membership = datasample(1:n_true_cluster, P, 'weights', prop_memb);
%    sigma_sq_beta_true = zeros(P, 1);
%    beta_true = zeros(P, 1);
%    for p = 1:P
        % Generate corresponding beta values
 %       sigma_sq_beta_true(p) = beta_sd(true_cluster_membership(p))^2;
 %       beta_true(p) = normrnd(beta_mu(true_cluster_membership(p)), sqrt(sigma_sq_beta_true(p))) ;
 %   end
%end

%beta_true = [ones(10,1)'*1.1, zeros(P-10,1)']';

%%Generate subject level error terms
% ei = normrnd(0.0, sqrt(true_sigma_sq), [N, 1]);
% sigX = zeros(P,P);
% for i=1:(P-1)       
%     sigX(i,i) = 1;
%     for j = (i+1):P
%         sigX(i,j) = 0.8^(j-i);
%         sigX(j,i) = sigX(i,j);
%     end
% end
% sigX(P,P) = 1;
% for i=1:N
% X(i,:) = mvnrnd(zeros(P,1),sigX);
% end
%%Generate the observed data
%Yi = zeros(N, 1);
%mui = zeros(N, 1);
%for i = 1:N
%    mui(i) = X(i, :) * beta_true + ei(i);
%end
%muicdf = normcdf(mui)';
%Yi(muicdf>0.5) = 1;
%Yi(muicdf <= 0.5) = -1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Posterior Computation Code is below

% Quantities needed for beta update
XtY = X' * Yi;
XtX = X' * X;

% Initialize error variance to random U(0, 1)
sigma_sq_epsilon = rand(1);

% Initialize sigma squared beta to random U(0, 1)
sigma_sq_beta = rand(P, 1);

% Storage/Posterior Quantities
beta_posterior_mean = zeros(P, 1);
beta_store          = zeros(P, nMCMC);
beta                = 2*(rand(P, 1) - 0.5);
sigma_sq_beta_posterior_mean = zeros(P, 1);

% Parameters related to slice sampler
atoms                          = rand(P, 1); % 2N clusters seems plenty, starting values uniform on (, 1)
sticks                         = cumprod(rand(2*P, 1), 1);
cm                             = reshape(datasample(1:n_start_cluster, P), [P,1]);
u_p                            = rand(P, 1) .* sticks(cm);
unnormalized_probabilities     = zeros(2*P, 1);
lambda_sq_posterior_sum        = zeros(2*P, 1);
cluster_number_samples         = zeros(2*P, 1);

% NEW Parameters
rho = zeros(N,1);
rho_store = zeros(N,nMCMC);

akap =0.1; bkap =1 ;

for iMCMC = 1:(nMCMC + nBurnin)
    
    % Print some status information
    if mod(iMCMC, 100) == 0
        disp(['Iteration: ' num2str(iMCMC)])
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
yest = X*beta;
%rho 
for sub=1:N
     rhoi_mu = abs(1-Yi(sub)*yest(sub))^(-1);
     if rhoi_mu == Inf
      rhoi_mu = 1e2; 
     end
   rhoi = random('InverseGaussian',rhoi_mu,sigma_sq_epsilon,1,1);
   rho(sub) = inv(rhoi);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Update the stick breaking weights
    min_u = min(u_p);
    % shouldnt be more than 2P clusters at extreme
    stick_prod      = 1.0;
    stick_remaining = 1.0;
    total_n         = P;
    for iclust = 1:(2*P)
        total_n           = total_n - sum(cm == iclust);
        raw_weight        = betarnd(1 + sum(cm == iclust), Conc + total_n);
        sticks(iclust)    = raw_weight * stick_prod;
        stick_remaining   = stick_remaining - sticks(iclust);
        stick_prod        = stick_prod * (1.0 - raw_weight);
        % If this is a new cluster then draw a new atom
        if sum(cm == iclust) == 0
            atoms(iclust) = gamrnd(r_lambda, 1 / delta_lambda);
        end
        % See if we can break early
        if stick_remaining < min_u
            nClust = iclust;
            break;
        end
    end

    % sample the u terms
    u_p = rand(P, 1) .* sticks(cm);

    % sample the cluster memberships
    cm_old = cm; % used to map cluster cleanup that follows
    for p = 1:P
        unnormalized_probabilities(:) = 0.0;
        for iClust = 1:nClust
            if u_p(p) < sticks(iClust)
                unnormalized_probabilities(iClust) = atoms(iClust)/2 * exp( -atoms(iClust)/2 * sigma_sq_beta(p) );
            end
        end
        normalized_probabilities = unnormalized_probabilities / sum(unnormalized_probabilities);

        % determine cluster membership
        cdraw = rand();
        sumprob = 0.0;
        for iClust = 1:nClust
            sumprob = sumprob + normalized_probabilities(iClust);
            if cdraw < sumprob
                cm(p) = iClust;
                break;
            end
        end
    end
    
    %% Bookkeeping the cluster information - no sampling in this block, just cleanup
    % First determine the index mapping for the clusters, skipping over
    % clusters which have no members
    max_cluster_index    = max(cm);
    actual_cluster_count = length(unique(cm));
    mappings             = zeros(actual_cluster_count, 1);
    current_cluster      = 0;
    for iClust = 1:(2*P)
        if sum(cm == iClust) > 0
            current_cluster = current_cluster + 1;
            mappings(current_cluster) = iClust; 
        end
    end
    % Now take care of the reassignment
    for i = 1:actual_cluster_count
        if i ~= mappings(i)
            atoms(i) = atoms(mappings(i));
            cm(cm == mappings(i)) = i;
            lambda_sq_posterior_sum(i) = lambda_sq_posterior_sum(mappings(i));
            cluster_number_samples(i) = cluster_number_samples(mappings(i));
        end
    end
    atoms((actual_cluster_count+1):end) = -Inf; 
    lambda_sq_posterior_sum((actual_cluster_count+1):end) = 0;
    cluster_number_samples((actual_cluster_count+1):end) = 0;
            
    %% Sample Beta
    % Now draw beta variance term conditioned on cluster membership
    for p = 1:P
        lambdaprime =  atoms(cm(p));
        muprime = sqrt( lambdaprime / beta(p)^2);
        sigma_sq_beta(p) = 1 / random('InverseGaussian',muprime,lambdaprime,1,1);
    end
    
    % Now update the betas    replace X by Xy and replace y by rho+1
    nXforbeta = sqrt(sigma_sq_epsilon)*diag(Yi./sqrt(rho))*X;
    nYforbeta = sqrt(sigma_sq_epsilon)*(rho+1)./sqrt(rho);
    nXtX = nXforbeta' * nXforbeta;
    nXtY = nXforbeta' * nYforbeta;
        posterior_variance = inv(nXtX + diag(1 ./ sigma_sq_beta));
        posterior_variance = (posterior_variance + posterior_variance') ./ 2;
        posterior_mean     = posterior_variance * (nXtY);
        % can be done more efficient if needed
        beta(:)            = mvnrnd(posterior_mean, posterior_variance);

%     
%     % Now update the betas    replace X by Xy and replace y by rho+1
%     posterior_variance = inv(XtX + diag(1 ./ sigma_sq_beta));
%     posterior_variance = (posterior_variance + posterior_variance') ./ 2;
%     posterior_mean     = posterior_variance * (XtY);
%     % can be done more efficient if needed
%     beta(:)            = mvnrnd(posterior_mean, sigma_sq_epsilon * posterior_variance);

    % Update the residual variance    - replace by \kappa update
    % (sigma_sq_epsilon)
    resid = ((rho+1) - diag(Yi)*X * beta)./sqrt(rho);
    sigma_sq_epsilon =  gamrnd( akap + (N-1)/2 , bkap + 0.5/ (resid'*resid) );
   % sigma_sq_epsilon = 0.5;
    % Update the atoms (lambda ^2 terms)
    for iClust = 1:nClust
        shape = sum(cm == iClust) + r_lambda;
        rate  = delta_lambda + sum( sigma_sq_beta(cm == iClust) )/2;
        atoms(iClust) = gamrnd( shape, 1 / rate );
    end

    if iMCMC > nBurnin
        beta_posterior_mean = beta_posterior_mean + beta./nMCMC;
        beta_store(:, iMCMC - nBurnin) = beta;
        rho_store(:,iMCMC-nBurnin)= rho;
        sigma_sq_beta_posterior_mean = sigma_sq_beta_posterior_mean + sigma_sq_beta ./ nMCMC;
        for c = 1:max(cm)
            lambda_sq_posterior_sum(c) = lambda_sq_posterior_sum(c) + atoms(c);
            cluster_number_samples(c)  = cluster_number_samples(c) + 1;
        end
    end
    [iMCMC, sigma_sq_epsilon]
end

% Final formatting
final_n_cluster = max(cm);
lambda_sq_posterior_mean = lambda_sq_posterior_sum(1:final_n_cluster) ./ cluster_number_samples(1:final_n_cluster);


%% Post-processing Analysis

Yest_te = Xtest*beta_posterior_mean;

muiest_te = Yest_te;
muiest_te(muiest_te >0) = 1;
muiest_te(muiest_te <= 0) = -1;
sum(abs(Ytest-muiest_te));
%misclassification error rate for test samples
misc_err_rate = sum((sign(Ytest)~=sign(muiest_te))/Ntest)

end


