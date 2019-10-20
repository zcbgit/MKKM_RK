function demo_mkkm_rk2_flowers102(n,nRepeat)
addpath('./MKKM_RK/');
addpath('./lib');
addpath('./dataset/flowers102');
datasets={'flowers102_10','flowers102_20','flowers102_30','flowers102_40','flowers102_50',...
    'flowers102_60','flowers102_70','flowers102_80','flowers102_90','flowers102_100'};
dataset=datasets{n};
load(dataset,'labels');
y=labels;
nClass=length(unique(y));
kernel_list={'Dhog','Dhsv','Dsiftbdy','Dsiftint'};
result_dir=fullfile(pwd,['result_mkkm_rk2_' num2str(nRepeat)],[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=zeros(length(y),length(y),nKernel);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
    K=exp(-mu*K);
    K=KernelNormalize(K,'Sample-Scale');
    Ks(:,:,iKernel)=K;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('MKKM with Representative Kernels begin ...');
lambdas=2.^(-15:5);
for j = 1:length(lambdas)
    lambda = lambdas(j);
    mkkm_rk_result = zeros(nRepeat,3);
    obj_final = cell(nRepeat,1);
    kw_aio = cell(nRepeat,1);
    Y_final = cell(nRepeat,1);
    suffix = num2str(lambda);
    result_file = fullfile(result_dir,[dataset,'_mkkm_rk2_' suffix '.mat']);
    rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['MKKM_RK ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
        Y = rand(size(Ks,3));
        [~, idx] = max(Y,[],2);
        Y = zeros(size(Y));
        Y(sub2ind(size(Y),[1:size(Y,1)]',idx)) = 1;
        [label,kw,obj,Y] = MKKM_RK2(Y,nClass,1e-5,Ks,lambda);
        mkkm_rk_result(iRepeat,:) = ClusteringMeasure(y,label);
        obj_final{iRepeat} = obj;
        kw_aio{iRepeat} = kw;
        Y_final{iRepeat} = Y;
        t_end = clock;
        disp(['MKKM_RK ',num2str(iRepeat),' of ' num2str(nRepeat),' iterations done.']);
        disp(['MKKM_RK exe time: ',num2str(etime(t_end, t_start))]);
    end
    if size(mkkm_rk_result,1) > 1
%         [~,minIdx] = min(obj_final{1});
%         mkkm_rk_result_obj = mkkm_rk_result(minIdx,:);
%         kw_obj = kw_aio{iRepeat};
        mkkm_rk_result_mean = mean(mkkm_rk_result);
    else
        mkkm_rk_result_mean = mkkm_rk_result;
    end
    save(result_file,'mkkm_rk_result','mkkm_rk_result_mean','kernel_list',...
        'kw_aio','obj_final','Y_final');
    disp('MKKM_RK on multi kernel done');
end