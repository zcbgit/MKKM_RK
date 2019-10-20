function demo_a_mkkm_caltech101(n,nRepeat)
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
result_dir=fullfile(pwd,['result_a_mkkm_' num2str(nRepeat)],[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ka = zeros(length(y));
result_file = fullfile(result_dir,[dataset,'_a_mkkm.mat']);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
%     mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
%     K=exp(-mu*K);
%     K=KernelNormalize(K,'Sample-Scale');
    Ka = Ka + K / nKernel;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp(['KernelKmeans on equal weighted multi kernel begin ...']);
t_start = clock;
a_mkkm_result = KernelKmeans_single_kernel(Ka, y, [], nRepeat);
t_end = clock;
runtime=etime(t_end, t_start);
runtime_mean=etime(t_end, t_start)/nRepeat;
disp(['A-MKKM exe time: ', num2str(etime(t_end, t_start))]);
if size(a_mkkm_result, 1) > 1
    a_mkkm_result_mean = mean(a_mkkm_result);
else
    a_mkkm_result_mean = a_mkkm_result;
end
save(result_file, 'a_mkkm_result', 'a_mkkm_result_mean', 'kernel_list',...
    'runtime','runtime_mean');
disp('A-MKKM on multi kernel done');