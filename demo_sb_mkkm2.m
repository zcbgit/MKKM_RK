function demo_sb_mkkm2(n, nRepeat)
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

result_dir = fullfile(pwd,['result_sb_mkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);
result_file = fullfile(result_dir,[dataset,'_sb_mkkm.mat']);
result_sb_mkkm = [];
sb_mkkm_result=zeros(nRepeat,3);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    if n~=3
        mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
        K=exp(-mu*K);
        K=KernelNormalize(K,'Sample-Scale');
    end
    iFile=[dataset '_' kernel_list{iKernel}];
    disp(['KernelKmeans on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)),' Kernel(', iFile, ') begin ...']);
    
    kkm_result_file = fullfile(result_dir, [iFile, '_result_kkm.mat']);
    
    t_start = clock;
    kkm_result = KernelKmeans_single_kernel(K, labels, fullfile(result_dir, iFile(1:end-4)), nRepeat);
    t_end = clock;
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
save(result_file, 'result_sb_mkkm', 'kernel_list','sb_mkkm_result','sb_mkkm_result_mean');
end