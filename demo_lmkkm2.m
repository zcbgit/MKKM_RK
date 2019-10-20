function demo_lmkkm2(n, nRepeat)
addpath('./LMKKM/');
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

result_dir = fullfile(pwd,['result_lmkkm_' num2str(nRepeat)],[dataset,'_result']);
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

kw_aio = cell(nRepeat,1);

disp('LMKKM begin ...');
lmkkm_result = zeros(nRepeat,3);
obj_final = zeros(nRepeat,1);
result_file = fullfile(result_dir,[dataset,'_lmkkm.mat']);
rng('default');
%set the number of iterations
parameters.cluster_count=nClass;
parameters.iteration_count = 100;
for iRepeat = 1:nRepeat
    t_start = clock;
    disp(['LMKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
    state = lmkkmeans_train(Ks, parameters);
    %     [label,kw,obj] = MKKM_RK(Y,nClass,1e-5,Ks);
    obj=state.objective(end);
    kw=state.Theta;
    pred_label=state.clustering;
    lmkkm_result(iRepeat,:) = ClusteringMeasure(labels,pred_label);
    obj_final(iRepeat) = obj;
    kw_aio{iRepeat} = kw;
    t_end = clock;
    disp(['LMKKM ',num2str(iRepeat),' of ' num2str(nRepeat),' iterations done.']);
    disp(['LMKKM exe time: ',num2str(etime(t_end, t_start))]);
end
if size(lmkkm_result,1) > 1
%     [~,minIdx] = min(obj_final);
%     lmkkm_result_obj = lmkkm_result(minIdx,:);
%     kw_obj = kw_aio{iRepeat};
    lmkkm_result_mean = mean(lmkkm_result);
else
    lmkkm_result_mean = lmkkm_result;
end
save(result_file,'lmkkm_result','lmkkm_result_mean','kernel_list','kw_aio');
disp('LMKKM on multi kernel done');