%% 
% load wine dataset which is in csv format; 
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

% visualization;

figure;
scatter(pdata(y==1,1),pdata(y==1,2), 'r'); hold on;
scatter(pdata(y==2,1),pdata(y==2,2), 'b'); 
scatter(pdata(y==3,1),pdata(y==3,2), 'g'); 


%%
% histogram for first dimension of pdata; 
% find the range of the data; 
datano = size(pdata, 1); 

min_data = min(pdata(:,1)); 
max_data = max(pdata(:,1)); 
nbin = 10; 
sbin = (max_data - min_data) / nbin; 
% create the bins; 
boundary = (min_data:sbin:max_data);

% just loop over the data points, and count how many of data points are in
% each bin; 
myhist = zeros(nbin, 1); 
for i = 1:datano
    which_bin = max(find(pdata(i,1) > boundary));
    myhist(which_bin) = myhist(which_bin) + 1; 
end
myhist = myhist * nbin ./ datano; 

%bar chart;
figure;
bar(boundary(1: end-1)+0.5 * sbin, myhist, 1.2*sbin);


%%
% for 2 dimensional data; 

min_data = min(pdata, [], 1); 
max_data = max(pdata, [], 1); 
nbin = 30; % you can change the number of bins in each dimension; 
sbin = (max_data - min_data) ./ nbin; 
% boundary = (min_data:sbin:max_data);
% create the bins; 
boundary = [min_data(1):sbin(1):max_data(1); min_data(2):sbin(2):max_data(2)]

% just loop over the data points, and count how many of data points are in
% each bin; 
myhist2 = zeros(nbin, nbin);
for i = 1:datano
    which_bin1 = max(find(pdata(i,1) > boundary(1,:)));
    which_bin2 = max(find(pdata(i,2) > boundary(2,:)));
    myhist2(which_bin1, which_bin2) = myhist2(which_bin1, which_bin2) + 1; 
end
myhist2 = myhist2 * nbin ./ datano; 

% two dimensional bar chart;
figure;
bar3(myhist2);

%% 
% kernel density estimator; 

% create an evaluation grid; 
gridno = 40; 
inc1 = (max_data(1) - min_data(1)) / gridno; 
inc2 = (max_data(2) - min_data(2)) / gridno; 
[gridx,gridy] = meshgrid(min_data(1):inc1:max_data(1), min_data(2):inc2:max_data(2)); 

% reshape everything to fit in one matrix;
gridall = [gridx(:), gridy(:)];     
gridallno = size(gridall, 1); 

norm_pdata = sum(pdata.^2, 2); 
norm_gridall = sum(gridall.^2, 2); 
cross = pdata * gridall'; 

% compute squared distance between each data point and the grid point; 
dist2 = repmat(norm_pdata, 1, gridallno) + repmat(norm_gridall', datano, 1) ...
    - 2 * cross; 

% choose kernel bandwidth 1; please also experiment with other bandwidth; 
bandwidth = 1; 
% evaluate the kernel function value for each training data point and grid
% point; 
kernelvalue = exp(-dist2 ./ bandwidth.^2); 

% sum over the training data point to the density value on the grid points;
% here I dropped the normalization factor in front of the kernel function,
% and you can add it back. It is just a constant scaling; 
mkde = sum(kernelvalue, 1) ./ datano; 

% reshape back to grid; 
mkde = reshape(mkde, gridno+1, gridno+1); 

% plot density as surface;
figure;
surf(gridx, gridy, mkde);































