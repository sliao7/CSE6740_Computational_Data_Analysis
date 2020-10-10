% this script is for 6740 2020Fall, HW1 Q2

clear; close all;
K = [2, 4, 8,16];
m = length(K);
figure;

for ii = 1:m
    [im1, im2] = Q2('beach.bmp', K(ii));
    subplot(m, 2, (ii-1)*2+1)
    h = imshow(im1, 'InitialMag',100, 'Border','tight');
	title(['K-medoids K=', num2str(K(ii))])
    subplot(m, 2, (ii-1)*2+2)
    h = imshow(im2, 'InitialMag',100, 'Border','tight');
	title(['K-means K=', num2str(K(ii))])
end