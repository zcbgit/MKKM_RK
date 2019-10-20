function demo_lmkkm_caltech101(n,nRepeat)
addpath('./LMKKM/');
addpath('./lib');
addpath('./dataset/caltech101');
datasets={'caltech101_10','caltech101_20','caltech101_30','caltech101_40','caltech101_50',...
    'caltech101_60','caltech101_70','caltech101_80','caltech101_90','caltech101_100'};
dataset=datasets{n};
load(dataset,'labels');
y=labels;
nClass=length(unique(y));
kernel_list={'echi2_phowColor_L0','echi2_phowColor_L1','echi2_phowColor_L2',...
    'echi2_phowGray_L0','echi2_phowGray_L1','echi2_phowGray_L2','echi2_ssim_L0',...
    'echi2_ssim_L1','echi2_ssim_L2','el2_gb'};
result_dir=fullfile(pwd,['result_lmkkm_' num2str(nRepeat)],[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=zeros(length(y),length(y),nKernel);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
%     mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
%     K=exp(-mu*K);
%     K=KernelNormalize(K,'Sample-Scale');
    Ks(:,:,iKernel)=K;
end

kw_aio = cell(nRepeat,1);
disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('LMKKM begin ...');
lmkkm_result = zeros(nRepeat,3);
obj_final = zeros(nRepeat,1);
result_file = fullfile(result_dir,[dataset,'_lmkkm.mat']);
rng('default');
%set the number of iterations
parameters.cluster_count=nClass;
parameters.iteration_count = 100;
runtime=zeros(nRepeat,1);
for iRepeat = 1:nRepeat
    t_start = clock;
    disp(['LMKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
    state = lmkkmeans_train(Ks, parameters);
    obj=state.objective(end);
    kw=state.Theta;
    label=state.clustering;
    lmkkm_result(iRepeat,:) = ClusteringMeasure(y,label);
    obj_final(iRepeat) = obj;
    kw_aio{iRepeat} = kw;
    t_end = clock;
    runtime(iRepeat)=etime(t_end, t_start);
    disp(['LMKKM ',num2str(iRepeat),' of ' num2str(nRepeat),' iterations done.']);
    disp(['LMKKM exe time: ',num2str(etime(t_end, t_start))]);
end
if size(lmkkm_result,1) > 1
    [~,minIdx] = min(obj_final);
    lmkkm_result_obj = lmkkm_result(minIdx,:);
    kw_obj = kw_aio{iRepeat};
    lmkkm_result_mean = mean(lmkkm_result);
    runtime_mean=mean(runtime);
else
    lmkkm_result_mean = lmkkm_result;
    runtime_mean=runtime;
end
save(result_file,'lmkkm_result','lmkkm_result_mean','kernel_list','kw_aio',...
    'runtime','runtime_mean');
disp('LMKKM on multi kernel done');