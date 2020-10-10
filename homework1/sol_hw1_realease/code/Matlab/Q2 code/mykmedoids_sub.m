function [ class, centroid ] = mykmedoids_sub( pixels_full, K)
% % this script is for 6740 2020Fall, HW1 Q2
% using sub-sampling 10% points from all raw data to fit kmedoids for simplification
%%
lp = 1; % distance type

nsample = size(pixels_full,1);

subsample_level = 0.1; % the ratio of subsample 
nds= round(subsample_level*nsample); % size of sub-sample points

pixels = datasample(pixels_full, nds); % subset of the data for training


c0=datasample(pixels,K);
c1=c0-10;

%%
dist_sub = zeros(nds, K); % all pair-wise distance between data and each centroids

old_cost = Inf;

list_empty = zeros(1, K); % to record if the any cluster is empty
z=1;
%%
while (norm(c1 - c0, 'fro') > 1e-6)
%     fprintf(1, '--iteration %d \n', z);
    
    %% assign the cluster label   
    for kk = 1:K % iterate clusters
        dist_sub(:, kk) = vecnorm(pixels- repmat(c0(kk, :), nds, 1), lp, 2);
    end
    
    [~, label] = min(dist_sub, [], 2);  % find the label for each node
    
    %% update the centroids
    new_cost = 0;
    for kk = 1:K
        idx_k = find(label ==kk); % the index of data in kth-cluster
        if isempty(idx_k)
            c1(kk,:) = Inf; % if kth-cluster is empty, then remove this cluster
            list_empty(kk) = 1;
        else
            nkk = length(idx_k); % total number of point in cluster k

            pix_k = pixels(idx_k, :); % all the points in cluster k
            cost_k = zeros(nkk, 1);  % to record the cost for each point as centroid

            for nn = 1:length(idx_k)  % iterate all point in kth-cluster
                cost_k(nn) = sum(vecnorm(pix_k - repmat(pix_k(nn,:),nkk,1), lp, 2) );
            end

            [t1, t2] = min(cost_k);
            new_cost = new_cost + t1; % record the cost for kth-cluster 
            c1(kk, :) = pixels(idx_k(t2),:); % update the centroid
        end
    end    
    if old_cost <= new_cost
        break;
    end
    c0=c1;
    old_cost = new_cost;
    z = z + 1;     
end   
fprintf(1, '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');
fprintf(1, 'Kmedoids, K= %d \n', K);
fprintf(1, 'number of empty clusters %d \n', sum(list_empty));
fprintf(1, '# of iterations %d \n', z);

%% determine the labels for all data points.
dist_all = zeros(nsample, K);
for kk = 1:K
    dist_all(:, kk) = vecnorm(pixels_full - repmat(c1(kk,:), nsample, 1), lp, 2);
end

[~, P1] = min(dist_all, [], 2);
    
class=P1;
centroid=c1;
end