%% load data
load('DATA.mat')
% Omega : N * Q matrix of network information
% scov  : N * C matric of covariates 
% Yi    : N * 1 matrix of label

% N : total number of subjects
% Q : total number of screened edges per subject
% C : total number of covariates per subject
% P : total number of varialbes (number of screened edges + number of covariates)


%% run HCP_LR1

[misc_err_rate,beta_posterior_mean,samp1,beta_store] = DPLSVM(Omega,scov,Ylabel)

% misc_err_rate: misclassificaiton rate based on posterior mean
% beta_posterior_mean: mean beta after burnin
% samp1: ids of subjects used in trainig set
% beta_store: saved beta after burnin 

%% save result
name1 = strcat('DPLSVM_result.mat');

save(name1,'misc_err_rate','beta_posterior_mean','samp1','beta_store');
