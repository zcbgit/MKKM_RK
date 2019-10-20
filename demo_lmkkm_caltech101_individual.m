function demo_lmkkm_caltech101_individual(m)
addpath('./LMKKM/');
addpath('./lib');
addpath('./dataset/caltech101');
datasets={'caltech101_10','caltech101_20','caltech101_30','caltech101_40','caltech101_50',...
    'caltech101_60','caltech101_70','caltech101_80','caltech101_90','caltech101_100'};
dims=[length(datasets),10];
index=indexTransRowwise(m,dims);
n=index(1);
iRepeat=index(2);
dataset=datasets{n};
load(dataset,'labels');
y=labels;
nClass=length(unique(y));
kernel_list={'echi2_phowColor_L0','echi2_phowColor_L1','echi2_phowColor_L2',...
    'echi2_phowGray_L0','echi2_phowGray_L1','echi2_phowGray_L2','echi2_ssim_L0',...
    'echi2_ssim_L1','echi2_ssim_L2','el2_gb'};
result_dir=fullfile(pwd,'result_lmkkm_individual',[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=zeros(length(y),length(y),nKernel);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    Ks(:,:,iKernel)=K;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('LMKKM begin ...');
result_file = fullfile(result_dir,[dataset,'_lmkkm_individual_' num2str(iRepeat) '.mat']);
rng('default');
%set the number of iterations
parameters.cluster_count=nClass;
parameters.iteration_count = 100;
%%
t_start = clock;
disp(['LMKKM ',num2str(iRepeat),' iteration begin ...']);
state = lmkkmeans_train(Ks, parameters);
obj_final=state.objective(end);
kw_aio=state.Theta;
label=state.clustering;
lmkkm_result = ClusteringMeasure(y,label);
t_end = clock;
disp(['LMKKM ',num2str(iRepeat),' iterations done.']);
disp(['LMKKM exe time: ',num2str(etime(t_end, t_start))]);
%%
save(result_file,'lmkkm_result','kernel_list','kw_aio');
disp('LMKKM on multi kernel done');