function [f, weight, dx]=AASC(w,cluster)
% Input
%       w : N x N x k affinity matrices 
% cluster : desired number of clusters
% Output
%      dx : clustering result
%  weight : weight assignment to affinity matrices

    sizeK=size(w,3);
    s1=size(w,1);
    s2=size(w,2);
    w_n=zeros(s1,s2);
    D_k=zeros(s1,s2,sizeK);
    L_k=zeros(s1,s2,sizeK);
    %initial equal weight
    weight=ones(1,sizeK)/sizeK;    
    for i=1:sizeK
        w_n=w_n+w(:,:,i)*(weight(1,i)^2);
        %compute D of each kernel
        s=sum(w(:,:,i),2);
        D_k(:,:,i)=diag(s);
        %compute Laplacian of each kernel
        L_k(:,:,i)=D_k(:,:,i)-w(:,:,i);
    end
    
    threshold=1e-5;
    iter_number=20;
    for iter=1:iter_number
        if iter>1
            f_old=f;
        end
        %%%%% find new f %%%%%
        if iter>1
            w_n=zeros(s1,s2);
            for i=1:sizeK
                w_n=w_n+w(:,:,i)*(weight(1,i)^2);
            end
        end
        %%% compute Laplacian matrix %%%
        D=diag(sum(w_n,1));
        L=D-w_n;
        %%% eigen decomposition %%%
        warning('off', 'MATLAB:eigs:NotAllEigsConverged');
        warning('off', 'MATLAB:eigs:NoEigsConverged');
        %OPTS.disp = 0;
        %[f, D_] = eigs((L+L')/2, D, cluster, 'SA', OPTS);%generalized eigenproblem
        L = (L+L')/2;
        L = full(L);
        [eigvec, eigval] = eig(L);
        f = eigvec(:, 1:min(cluster, size(eigvec,2)));
        %check termination condition
        if iter > 1
            tmp=abs(f)-abs(f_old);
            tmp=tmp.^2;
            if sum(sum(tmp))<threshold
                break;
            end
        end
        
        %%%%% find new weight %%%%%
        alpha_k=zeros(cluster,sizeK);
        beta_k=zeros(cluster,sizeK);
        gamma_k=zeros(cluster,sizeK);
        %calculate coefficients alpha_k, beta_k, gamma_k
        for k=1:sizeK
            %compute alpha
            tmp=diag(f'*D_k(:,:,k)*f);
            alpha_k(:,k)=tmp;
            %compute beta
            tmp=diag(f'*L_k(:,:,k)*f);
            beta_k(:,k)=tmp;
            %compute gamma
            gamma_k(:,k)=beta_k(:,k)./alpha_k(:,k); 
        end
        %%%-------------------- 1-D solution ----------------------------------
        alpha=alpha_k(2,:);
        gamma=gamma_k(2,:);
        weight_1D=zeros(sizeK+2,sizeK);
        result_1D=zeros(sizeK+2,1);
        %find optimal
        for iter_1=0:1:sizeK+1 
            %lambda_1
            lambda_1 = solve_lambda(alpha,gamma,iter_1);
            %lambda_2
            a=ones(1,sizeK);
            b=(gamma-lambda_1).*alpha;
            lambda_2=1/sum(a./b);
            %u_k
            c=sqrt(alpha).*(gamma-lambda_1);
            u=a.*lambda_2;
            u=u./c;
            %v_k
            weight_1D(iter_1+1,:)=u./sqrt(alpha);
        end
        [B,IX] = sort(result_1D);
        weight=weight_1D(IX(1),:);
    end
	if nargout > 2
		if ~isempty(which('litekmeans.m'))
			dx = litekmeans(f, cluster, 'replicates', 20);
		else
			dx = kmeans(f,cluster,'EmptyAction','drop','Replicates',50);
		end
	end
    clear w_n;
    clear D_k;
    clear L_k;
    clear alpha_k;
    clear beta_k;
    clear gamma_k;
    clear alpha;
    clear gamma;
    % clear f;
    clear weight_1D;
    clear result_1D;
    clear D;
    clear L;
   