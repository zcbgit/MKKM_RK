function demo_a_mkkm2(n, nRepeat)
addpath('./KernelKmeans/');
addpath('./lib');
addpath('dataset')
datasets={'flowers17','flowers102','caltech101'};
dataset=datasets{n};

load(dataset,'labels');
nClass = length(unique(labels));
switch dataset
    case 'flowers17'
        kernel_list={'D_colourgc','D_hog','D_hsv','D_shapegc','D_siftbdy',...
            'D_siftint','D_texturegc'};
    case 'flowers102'
        kernel_list={'Dhog','Dhsv','Dsiftbdy','Dsiftint'};
    case 'caltech101'
        kernel_list={'echi2_phowColor_L0','echi2_phowColor_L1','echi2_phowColor_L2',...
            'echi2_phowGray_L0','echi2_phowGray_L1','echi2_phowGray_L2','echi2_ssim_L0',...
            'echi2_ssim_L1','echi2_ssim_L2','el2_gb'};
end

result_dir = fullfile(pwd,['result_a_mkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ka = zeros(length(labels));

for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    if n~=3
        mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
        K=exp(-mu*K);
        K=KernelNormalize(K,'Sample-Scale');
    end
    Ka = Ka + K / nKernel;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp(['KernelKmeans on equal weighted multi kernel begin ...']);
result_file = fullfile(result_dir,[dataset,'_a_mkkm.mat']);
t_start = clock;
a_mkkm_result = KernelKmeans_single_kernel(Ka, labels, [], nRepeat);
t_end = clock;
disp(['A-MKKM exe time: ', num2str(etime(t_end, t_start))]);
if size(a_mkkm_result, 1) > 1
    a_mkkm_result_mean = mean(a_mkkm_result);
else
    a_mkkm_result_mean = a_mkkm_result;
end
save(result_file, 'a_mkkm_result', 'a_mkkm_result_mean', 'kernel_list');
disp('A-MKKM on multi kernel done');