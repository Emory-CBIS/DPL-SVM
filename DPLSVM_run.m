%% load data
load('DATA.mat')
% Omega : input network information for subjects
% scov  : input supplementary covariates for subjects
% Ylabel: input label for subjects

%% run HCP_LR1

[misc_err_rate,beta_posterior_mean,samp1,beta_store] = DPLSVM(Omega,scov,Ylabel)

% misc_err_rate: misclassificaiton rate based on posterior mean
% beta_posterior_mean: mean beta after burnin
% samp1: ids of subjects used in trainig set
% beta_store: saved beta after burnin 

%% save result
name1 = strcat('DPLSVM_result.mat');

save(name1,'misc_err_rate','beta_posterior_mean','samp1','beta_store');
