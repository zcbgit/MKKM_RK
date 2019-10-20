function demo_sb_mkkm(n, nRepeat)
addpath('./KernelKmeans/');
addpath('./lib');
datasets={'AR_840n_768d_120c','AR_840n_768d_120c_uni','binaryalphadigs_1404n_320d_36c',...
    'COIL20_1440n_1024d_20c','jaffe_213n_676d_10c','jaffe_213n_676d_10c_uni',...
    'ORL_400n_1024d_40c','ORL_400n_1024d_40c_zscore_uni','tr11_414n_6429d_9c_tfidf_uni',...
    'tr41_878n_7454d_10c_tfidf_uni','tr45_690n_8261d_10c_tfidf_uni','YALE_165n_1024d_15c',...
    'YALE_165n_1024d_15c_zscore_uni'};
dataset=datasets{n};
data_dir = fullfile(pwd,'dataset');
kernel_dir = fullfile(pwd,'kernels',[dataset,'_kernel']);
file_list = dir(kernel_dir);

kernel_list = {};
iKernel = 0;
for iFile = 1:length(file_list)
    sName = file_list(iFile).name;
    if (~strcmp(sName,'.') && ~strcmp(sName,'..'))
        iKernel = iKernel+1;
        kernel_list{iKernel} = sName;
    end
end

load(fullfile(data_dir,dataset),'y');
nClass = length(unique(y));

result_dir = fullfile(pwd,['result_sb_mkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end


disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);
result_file = fullfile(result_dir,[dataset,'_sb_mkkm.mat']);
result_sb_mkkm = [];
sb_mkkm_result=zeros(nRepeat,3);
runtime_mean=0;
for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    clear K;
    load(fullfile(kernel_dir, iFile), 'K');
    
    disp(['KernelKmeans on ',  num2str(iKernel), ' of ' num2str(length(kernel_list)), ' Kernel(',  iFile(1:end-4), ') begin ...']);
    kkm_result_file = fullfile(result_dir, [iFile(1:end-4), '_result_kkm.mat']);
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
save(result_file, 'result_sb_mkkm', 'kernel_list','sb_mkkm_result','sb_mkkm_result_mean',...
    'runtime_mean');
end