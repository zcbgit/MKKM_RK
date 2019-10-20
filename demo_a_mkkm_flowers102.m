function demo_a_mkkm_flowers102(n,nRepeat)
addpath('./KernelKmeans/');
addpath('./lib');
addpath('./dataset/flowers102');
datasets={'flowers102_10','flowers102_20','flowers102_30','flowers102_40','flowers102_50',...
    'flowers102_60','flowers102_70','flowers102_80','flowers102_90','flowers102_100'};
dataset=datasets{n};
load(dataset,'labels');
y=labels;
kernel_list={'Dhog','Dhsv','Dsiftbdy','Dsiftint'};
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
    mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
    K=exp(-mu*K);
    K=KernelNormalize(K,'Sample-Scale');
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