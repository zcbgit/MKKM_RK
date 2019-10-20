function demo_a_mkkm(n, nRepeat)
addpath('./KernelKmeans/');
addpath('./lib');
datasets={'AR_840n_768d_120c_uni','binaryalphadigs_1404n_320d_36c',...
    'COIL20_1440n_1024d_20c','jaffe_213n_676d_10c_uni',...
    'ORL_400n_1024d_40c_zscore_uni','tr11_414n_6429d_9c_tfidf_uni',...
    'tr41_878n_7454d_10c_tfidf_uni','tr45_690n_8261d_10c_tfidf_uni',...
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

result_dir = fullfile(pwd,['result_a_mkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ka = zeros(length(y));
result_file = fullfile(result_dir,[dataset,'_a_mkkm.mat']);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    load(fullfile(kernel_dir,iFile),'K');
    Ka = Ka + K / nKernel;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp(['KernelKmeans on equal weighted multi kernel begin ...']);
t_start = clock;
a_mkkm_result = KernelKmeans_single_kernel(Ka, y, [], nRepeat);
t_end = clock;
runtime=etime(t_end, t_start)/nRepeat;
disp(['A-MKKM exe time: ', num2str(etime(t_end, t_start))]);
if size(a_mkkm_result, 1) > 1
    a_mkkm_result_mean = mean(a_mkkm_result);
else
    a_mkkm_result_mean = a_mkkm_result;
end
save(result_file, 'a_mkkm_result', 'a_mkkm_result_mean', 'kernel_list','runtime');
disp('A-MKKM on multi kernel done');