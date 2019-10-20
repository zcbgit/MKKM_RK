function demo_mkkm2(n, nRepeat)
addpath('./MKKM/');
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

result_dir = fullfile(pwd,['result_mkkm_' num2str(nRepeat)],[dataset,'_result']);
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

disp('MKKM begin ...');
mkkm_result = [];
obj_final = [];
result_file = fullfile(result_dir,[dataset,'_mkkm.mat']);
rng('default');
for iRepeat = 1:nRepeat
    t_start = clock;
    disp(['MKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
    U = rand(size(Ks,1), nClass);
    [~, uidx] = max(U, [], 2);
    U = zeros(size(U));
    U(sub2ind(size(U), [1:size(U,1)]', uidx)) = 1;
    [label_mkkm, kw, obj] = MKKM(U, nClass, 1, 1e-5, Ks);
    mkkm_result = [mkkm_result; ClusteringMeasure(labels, label_mkkm)];%#ok<AGROW>
    obj_final = [obj_final; obj];%#ok<AGROW>
    kw_aio{iRepeat} = kw;
    t_end = clock;
    disp(['MKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
    disp(['MKKM exe time: ', num2str(etime(t_end, t_start))]);
end
if size(mkkm_result, 1) > 1
%     [~, minIdx] = min(obj_final);
%     mkkm_result_obj = mkkm_result(minIdx,:);
%     kw_obj = kw_aio{iRepeat};
    mkkm_result_mean = mean(mkkm_result);
else
    mkkm_result_mean = mkkm_result;
end
save(result_file, 'mkkm_result', 'mkkm_result_mean', 'kernel_list', 'kw_aio');
disp('MKKM on multi kernel done');