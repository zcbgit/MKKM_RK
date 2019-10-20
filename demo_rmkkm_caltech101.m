function demo_rmkkm_caltech101(n,nRepeat)
addpath('./RMKKM/');
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
result_dir=fullfile(pwd,['result_rmkkm_' num2str(nRepeat)],[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=cell(nKernel,1);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
%     mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
%     K=exp(-mu*K);
%     K=KernelNormalize(K,'Sample-Scale');
    Ks{iKernel}=K;
end

kw_aio = cell(nRepeat,1);
disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('RMKKM begin ...');
gammaCandidates = (0.1:0.1:0.9);
for gammaIdx = 1:length(gammaCandidates)
    gamma = gammaCandidates(gammaIdx);
    rmkkm_result = zeros(nRepeat,3);
    obj_final = [];
    suffix = num2str(gamma);
    result_file = fullfile(result_dir,[dataset,'_rmkkm_' suffix '.mat']);
    runtime=zeros(nRepeat,1);
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['MKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
        [label_rmkkm, kw, ~, ~, obj] = RMKKM(Ks, nClass, 'gamma', gamma, 'maxiter', 50, 'replicates', 1);
        rmkkm_result(iRepeat,:) = ClusteringMeasure(y, label_rmkkm);
        obj_final = [obj_final; obj];%#ok<AGROW>
        kw_aio{iRepeat} = kw;
        t_end = clock;
        disp(['RMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['RMKKM exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(rmkkm_result, 1) > 1
        rmkkm_result_mean = mean(rmkkm_result);
        runtime_mean=mean(runtime);
    else
        rmkkm_result_mean = rmkkm_result;
        runtime_mean=runtime;
    end
    save(result_file, 'rmkkm_result', 'rmkkm_result_mean', 'kernel_list', 'kw_aio',...
        'runtime','runtime_mean');
    disp('RMKKM on multi kernel done');
end