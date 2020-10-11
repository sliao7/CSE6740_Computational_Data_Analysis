%% GMM, ISyE 6740, Xie
clear;
clc;
close all

data = csvread('wine.data'); 
y = data(:,1); 
data = data(:,2:end); 

%% pca the data;
[ndata, mu, sigma] = zscore(data); 
covariance = cov(ndata); 
d = 2; 
[V, S] = eigs(covariance, d); 

% project the data to the top 2 principal directions;
pdata = ndata * V;

datano = size(pdata, 1); 

min_data = min(pdata, [], 1); 
max_data = max(pdata, [], 1); 

%% 
% kernel density estimator; 

% create an evaluation grid; 
gridno = 40; 
inc1 = (max_data(1) - min_data(1)) / gridno; 
inc2 = (max_data(2) - min_data(2)) / gridno; 
[gridx,gridy] = meshgrid(min_data(1):inc1:max_data(1), min_data(2):inc2:max_data(2)); 

gridall = [gridx(:), gridy(:)];     
gridallno = size(gridall, 1); 

%%
% em algorithm for fitting mixture of gaussians; 

rng(1e5)
% fit a mixture of 3 gaussians; 
K = 3; 
% randomly initialize the paramters; 
% mixing proportion; 
pi = rand(K,1); 
pi = pi./sum(pi); 
% mean or center of gaussian; 
mu = randn(2, K); 
% covariance, and make sure it is positive semidefinite; 
sigma = zeros(2, 2, K); 
for i = 1:K
    tmp = randn(2, 2); 
    sigma(:,:,i) = tmp * tmp'; 
end
% poster probability of component indicator variable; 
tau = zeros(datano, K); 

% we just choose to run 100 iterations, but you can change the termination
% criterion for the loop to whether the solution changes big enough between
% two adjacent iterations; 
iterno = 100; 
figure; 
for it = 1:iterno
    fprintf(1, '--iteration %d of %d\n', it, iterno); 
    % alternate between e and m step; 
    
    % E-step; 
    for i = 1:K
        tau(:,i) = pi(i) * mvnpdf(pdata, mu(:,i)', sigma(:,:,i)); 
    end
    sum_tau = sum(tau, 2); 
    % normalize
    tau = tau ./ repmat(sum_tau, 1, K);
        
    % M-step; 
    for i = 1:K
        % update mixing proportion; 
        pi(i) = sum(tau(:,i), 1) ./ datano; 
        % update gaussian center; 
        mu(:, i) = pdata' * tau(:,i) ./ sum(tau(:,i), 1); 
        % update gaussian covariance;
        tmpdata = pdata - repmat(mu(:,i)', datano, 1); 
        sigma(:,:,i) = tmpdata' * diag(tau(:,i)) * tmpdata ./ sum(tau(:,i), 1); 
    end
    
    % plot data points using the mixing proportion tau as colors; 
    % the data point locations will not change over iterations, but the
    % color may change; 
    scatter(pdata(:,1), pdata(:,2), 16*ones(datano, 1), tau, 'filled'); 

    hold on; 
    % also plot the centers of the guassian; 
    % the centers change locations each iteraction until the solution converges;  
    scatter(mu(1,:)', mu(2,:)', 26*ones(K, 1), 'filled'); 
    drawnow; 
    
    % also draw the contour of the fitted mixture of gaussian density; 
    % first evaluate the density on the grid points; 
    tmppdf = zeros(size(gridall,1), 1);
    for i = 1:K        
        tmppdf = tmppdf + pi(i) * mvnpdf(gridall, mu(:,i)', sigma(:,:,i));
    end
    tmppdf = reshape(tmppdf, gridno+1, gridno+1); 
    
    % draw contour; 
    [c, h] = contour(gridx, gridy, tmppdf);
    
    hold off; 
    
    pause(0.1);

end





























