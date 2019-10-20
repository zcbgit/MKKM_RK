function demo_mkkm_caltech101(n,nRepeat)
addpath('./MKKM/');
addpath('./lib');
addpath('./dataset');
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
result_dir=fullfile(pwd,['result_mkkm_' num2str(nRepeat)],[dataset '_result']);
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

disp('MKKM begin ...');
mkkm_result = [];
obj_final = [];
result_file = fullfile(result_dir,[dataset,'_mkkm.mat']);
rng('default');
runtime=zeros(nRepeat,1);
for iRepeat = 1:nRepeat
    t_start = clock;
    disp(['MKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
    U = rand(size(Ks,1), nClass);
    [~, uidx] = max(U, [], 2);
    U = zeros(size(U));
    U(sub2ind(size(U), [1:size(U,1)]', uidx)) = 1;
    [label_mkkm, kw, obj] = MKKM(U, nClass, 1, 1e-5, Ks);
    mkkm_result = [mkkm_result; ClusteringMeasure(y, label_mkkm)];%#ok<AGROW>
    obj_final = [obj_final; obj];%#ok<AGROW>
    kw_aio{iRepeat} = kw;
    t_end = clock;
    runtime(iRepeat)=etime(t_end, t_start);
    disp(['MKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
    disp(['MKKM exe time: ', num2str(etime(t_end, t_start))]);
end
if size(mkkm_result, 1) > 1
    [~, minIdx] = min(obj_final);
    mkkm_result_obj = mkkm_result(minIdx,:);
    kw_obj = kw_aio{iRepeat};
    mkkm_result_mean = mean(mkkm_result);
    runtime_mean=mean(runtime);
else
    mkkm_result_mean = mkkm_result;
    runtime_mean=runtime;
end
save(result_file, 'mkkm_result', 'mkkm_result_mean', 'kernel_list', 'kw_aio',...
    'runtime','runtime_mean');
disp('MKKM on multi kernel done');