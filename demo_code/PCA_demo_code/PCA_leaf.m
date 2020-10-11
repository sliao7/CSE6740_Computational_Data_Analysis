%leaf dataset

clear
clc
close all

data=importdata('leaf.mat');

%dataset description
%The provided data comprises the following shape (attributes 3 to 9) and texture (attributes 10
%to 16) features:
%1. Class (Species)
%2. Specimen Number
%3. Eccentricity
%4. Aspect Ratio
%5. Elongation
%6. Solidity
%7. Stochastic Convexity
%8. Isoperimetric Factor
%9. Maximal Indentation Depth
%10. Lobedness
%11. Average Intensity
%12. Average Contrast
%13. Smoothness
%14. Third moment
%15. Uniformity
%16. Entropy

% In this example, we will normalize the data because the features have
% very different values 



% extract attributes from the raw data
Anew = data(:,3:end)';
data_std = std(Anew');
[n,m] = size(Anew);
Anew = bsxfun(@rdivide, Anew, data_std'); 
% create indicator matrix; 
Inew = data(:,1); 

% PCA
mu = mean(Anew')'; % mean
xc = bsxfun(@minus, Anew, mu); 
C = xc * xc'./ m; % covariance

k = 2; 
[W, S] = eigs(C,k); 

figure; subplot(2,1,1); stem(W(:,1),'r-o'); title('Largest Eigenvector')
subplot(2,1,2); stem(W(:,2),'r-o'); title('2nd Largest Eigenvector')

dim1 = W(:,1)' * xc ./ sqrt(S(1,1));
dim2 = W(:,2)' * xc ./ sqrt(S(2,2));

color_string = 'bgrmck'; 
marker_string = '.+*o';
figure; 
hold on; 
for i = 1:max(Inew)
  plot(dim1(Inew==i), dim2(Inew==i), [color_string(mod(i,5)+1), marker_string(mod(i,4)+1)]); 
end
hold off; 






