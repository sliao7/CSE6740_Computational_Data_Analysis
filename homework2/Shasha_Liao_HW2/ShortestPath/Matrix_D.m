function D = Matrix_D(W)
% A function to compute matrix D from weight matrix W defining the weight
% of edges between each pair of nodes.
% Note that you can assign sufficiently large weights to non-existing
% edges.

n = size(W,1);
N = 1:n;
One = ones(n,1);
J = kron(N, One);
I = J';
I = reshape(I,[1, n^2]);
J = reshape(J,[1, n^2]);

W_sym = zeros([n,n]);
for i=1:n
    for j = 1:n
        W_sym(i,j) = min(W(i,j), W(j,i));
    end
end
W_sym = reshape(W_sym, [1, n^2]);
G = digraph(I,J,W_sym);
D = distances(G);
return

