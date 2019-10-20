function demo_lmkkm2_individual(m)
addpath('./LMKKM/');
addpath('./lib');
addpath('dataset')
datasets={'flowers17','flowers102','caltech101'};
dims=[length(datasets),10];
index=indexTransRowwise(m,dims);
n=index(1);
iRepeat=index(2);
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

result_dir = fullfile(pwd,'result_lmkkm_individual',[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks = zeros(length(labels),length(labels),nKernel);

for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    if n~=3
        mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
        K=exp(-mu*K);
        K=KernelNormalize(K,'Sample-Scale');
    end
    Ks(:,:,iKernel) = K;
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
pred_label=state.clustering;
lmkkm_result = ClusteringMeasure(labels,pred_label);
t_end = clock;
disp(['LMKKM ',num2str(iRepeat),' iterations done.']);
disp(['LMKKM exe time: ',num2str(etime(t_end, t_start))]);
%%
save(result_file,'lmkkm_result','kernel_list','kw_aio');
disp('LMKKM on multi kernel done');