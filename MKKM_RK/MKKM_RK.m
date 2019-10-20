function [idx,w,objective,Y,H_normalized] = MKKM_RK(Y,cluster,error,kvalue,...
    measurement,lambda)
% Input
%   Y : M x M indicator matrix
%   cluster : desired number of clusters
%   error : stop threshold
%   kvalue : N x N x k affinity matrices
%   measurement : {'sfn','fip','vn','ld'};
%   lambda : tradeoff parameter of diversity regularization
% 
% Output
%   idx : N x 1 cluster indices of each observation
%   H_normalized : k-dimensional representations of the samples on the unit sphere
%   Y : indicator of kernel representation
%
iteration_count = 100;
n = size(kvalue, 1);
M = size(kvalue,3);

if isempty(error)
    error = 1e-5;
end
D = dissimilarity(kvalue, measurement);
w = mean(Y,2);
K_w = calculate_kernel_w(kvalue,w.^2);
objective = zeros(iteration_count,1);

% Main loop
for iter=1:iteration_count
    fprintf(1,'running iteration %d...\n',iter);
    % compute the membership matrix
    [H,~] = eigs(K_w,cluster);
    C = zeros(M,M);
    for i = 1:M
        C(i,i) = trace(kvalue(:,:,i)*(eye(n)-H*H'));
    end
    cvx_begin
        variable Y(M,M)
        minimize(1/M^2*(Y*ones(M,1))'*C*(Y*ones(M,1))+lambda*trace(D'*Y))
        subject to
            ones(1,M)*Y==ones(1,M);
            Y>=zeros(M);
    cvx_end
    w = mean(Y,2);
    K_w = calculate_kernel_w(kvalue,w.^2);
    objective(iter) = trace(K_w*(eye(n)-H*H'))+lambda*trace(D'*Y);    
    % check termination condition
    if iter > 1
        if abs(objective(iter) - objective(iter-1))< error
            break;
        end
    end
end
H_normalized = H ./ repmat(sqrt(sum(H.^2, 2)), 1, cluster);
idx = kmeans(H_normalized,cluster,'MaxIter',1000,'Replicates',10);
% obj_final = objective(end);

% compute the dissimilarity between basic kernels
function D = dissimilarity(K, measurement)
num_kernels = size(K,3);
D = zeros(num_kernels);
for i = 1:num_kernels
    Ki=K(:,:,i);
    for j = i:num_kernels
        Kj=K(:,:,j);
        switch measurement
            case 'sfn'
                D(i,j) = trace((Ki-Kj)'*(Ki-Kj));
                D(j,i) = D(i,j);
            case 'fip'
                D(i,j) = trace(Ki'*Kj);
                D(j,i) = D(i,j);
            case 'vn'
                D(i,j)=trace(Ki*logm(Ki)-Ki*logm(Kj)-Ki+Kj);
                D(j,i)=trace(Kj*logm(Kj)-Kj*logm(Ki)-Kj+Ki);
            case 'ld'
                temp=Ki*pinv(Kj);
                D(i,j)=trace(temp)-log(det(temp))-D;
                temp=Kj*pinv(Ki);
                D(j,i)=trace(temp)-log(det(temp))-D;
        end
    end
end

% compute the compositional kernel
function K_theta = calculate_kernel_w(K,theta)
K_theta = zeros(size(K(:,:,1)));
for m = 1:size(K,3)
    K_theta = K_theta+theta(m)*K(:,:,m);
end