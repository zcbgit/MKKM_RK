function demo_lmkkm_flowers102_individual(m)
addpath('./LMKKM/');
addpath('./lib');
addpath('./dataset/flowers102');
datasets={'flowers102_10','flowers102_20','flowers102_30','flowers102_40','flowers102_50',...
    'flowers102_60','flowers102_70','flowers102_80','flowers102_90','flowers102_100'};
dims=[length(datasets),10];
index=indexTransRowwise(m,dims);
n=index(1);
iRepeat=index(2);
dataset=datasets{n};
load(dataset,'labels');
y=labels;
nClass=length(unique(y));
kernel_list={'Dhog','Dhsv','Dsiftbdy','Dsiftint'};
result_dir=fullfile(pwd,'result_lmkkm_individual',[dataset '_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=zeros(length(y),length(y),nKernel);
for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
    K=exp(-mu*K);
    K=KernelNormalize(K,'Sample-Scale');
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