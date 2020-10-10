function [ class, centroid ] = mykmeans( pixels, K )
% % this script is for 6740 2020Fall, HW1 Q2

for k=1:K
c1(k,:)=quantile(pixels,(0.5/K+(k-1)/K));
end
c0=c1-10;

i=1;
while (norm(c1 - c0, 'fro') > 1e-6)
%     fprintf(1, '--iteration %d \n', i);
    
    % record previous c; 
    c0 = c1; 
    
    % assign data points to current cluster; 
    for j = 1:length(pixels) % loop through data points; 
        tmp_distance = zeros(1, K); 
        for k = 1:K % through centers; 
            tmp_distance(k) = sum((pixels(j,:) - c1(k,:)).^2); % norm(x(:,j) - c(:,k)); 
        end
        [~,K_index] = min(tmp_distance); % ~ ignores the first argument; 
        P(:, j) = zeros(K, 1); 
        P(K_index, j) = 1; 
    end
        
    % adjust the cluster centers according to current assignment;     
%     cstr = {'r.', 'b.', 'g.', 'r+', 'b+', 'g+'};
    obj = 0;
    obj2=0;
    for k = 1:K
        idx = find(P(k, :)>0); 
        no_of_points = length(idx);  
        if (no_of_points == 0) 
            % a center has never been assigned a data point; 
            % re-initialize the center; 
            c1(k,:) = quantile(pixels,0.5);  
        else
            % equivalent to sum(x(:,idx), 2) ./ no_of_points;            
            c1(k,:) = P(k,:) * pixels ./ no_of_points;         
        end
        obj = obj + sum(sum((pixels(idx,:) - repmat(c0(k,:),no_of_points,1)).^2));
        obj2 = obj2 + sum(sum((pixels(idx,:) - repmat(c1(k,:),no_of_points,1)).^2));
    end
    
    if obj2-obj>0
        c1=c0;
    end
    i = i + 1;     
end

fprintf(1, '\n ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~');
fprintf(1, 'Kmeans, K= %d \n', K);
fprintf(1, '# of iterations for Kmeans %d \n', i);

P1=(sum(P.*(1:K)'))';
    
    class=P1;
    centroid=c1;
end

