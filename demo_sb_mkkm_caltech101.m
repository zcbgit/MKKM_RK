function demo_sb_mkkm_caltech101(n,nRepeat)
addpath('./KernelKmeans/');
addpath('./lib');
addpath('./dataset/caltech101');
datasets={'caltech101_10','caltech101_20','caltech101_30','caltech101_40','caltech101_50',...
    'caltech101_60','caltech101_70','caltech101_80','caltech101_90','caltech101_100'};
dataset=datasets{n};
load(dataset,'labels');
y=labels;
kernel_list={'echi2_phowColor_L0','echi2_phowColor_L1','echi2_phowColor_L2',...
    'echi2_phowGray_L0','echi2_phowGray_L1','echi2_phowGray_L2','echi2_ssim_L0',...
    'echi2_ssim_L1','echi2_ssim_L2','el2_gb'};
result_dir=fullfile(pwd,['result_sb_mkkm_' num2str(nRepeat)],[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

result_sb_mkkm = [];
sb_mkkm_result=zeros(nRepeat,3);
result_file = fullfile(result_dir,[dataset,'_sb_mkkm.mat']);
runtime_mean=0;
for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
%     mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
%     K=exp(-mu*K);
%     K=KernelNormalize(K,'Sample-Scale');
    
    disp(['KernelKmeans on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)), ' Kernel(',  iFile(1:end-4), ') begin ...']);
    kkm_result_file = fullfile(result_dir, [iFile, '_result_kkm.mat']);
    t_start = clock;
    kkm_result = KernelKmeans_single_kernel(K, y, fullfile(result_dir, iFile(1:end-4)), nRepeat);
    t_end = clock;
    runtime_mean=runtime_mean+etime(t_end, t_start)/nRepeat;
    disp(['KernelKmeans exe time: ', num2str(etime(t_end, t_start))]);
    save(kkm_result_file, 'kkm_result');
    if mean(kkm_result(:,1))>mean(sb_mkkm_result(:,1))
        sb_mkkm_result(:,1)=kkm_result(:,1);
    end
    if mean(kkm_result(:,2))>mean(sb_mkkm_result(:,2))
        sb_mkkm_result(:,2)=kkm_result(:,2);
    end
    if mean(kkm_result(:,3))>mean(sb_mkkm_result(:,3))
        sb_mkkm_result(:,3)=kkm_result(:,3);
    end
    disp(['KernelKmeans on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)), ' Kernel(',  iFile(1:end-4), ') done']);
    if size(kkm_result, 1) > 1
        result_sb_mkkm = [result_sb_mkkm; mean(kkm_result)]; %#ok<AGROW>
    else
        result_sb_mkkm = [result_sb_mkkm; kkm_result]; %#ok<AGROW>
    end
    clear K kkm_result;
end
sb_mkkm_result_mean=mean(sb_mkkm_result);
save(result_file, 'result_sb_mkkm','kernel_list','sb_mkkm_result','sb_mkkm_result_mean',...
    'runtime_mean');
end