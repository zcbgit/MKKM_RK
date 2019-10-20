function demo_rmkkm2(n, nRepeat)
addpath('./RMKKM/');
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

result_dir = fullfile(pwd,['result_rmkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=cell(nKernel,1);

for iKernel = 1:length(kernel_list)
    load(dataset,kernel_list{iKernel});
    K=eval(kernel_list{iKernel});
    if n~=3
        mu=1/(mean(mean(K))); % refer to original paper 2018ICVGIP
        K=exp(-mu*K);
        K=KernelNormalize(K,'Sample-Scale');
    end
    Ks{iKernel}=K;
end

kw_aio = cell(nRepeat,1);
disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('RMKKM begin ...');
gammaCandidates = (0.1:0.1:0.9);
for gammaIdx = 1:length(gammaCandidates)
    gamma = gammaCandidates(gammaIdx);
    rmkkm_result = zeros(nRepeat,3);
    obj_final = [];
    suffix = num2str(gamma);
    result_file = fullfile(result_dir,[dataset,'_rmkkm_' suffix '.mat']);
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['MKKM ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
        [label_rmkkm, kw, ~, ~, obj] = RMKKM(Ks, nClass, 'gamma', gamma, 'maxiter', 50, 'replicates', 1);
        rmkkm_result(iRepeat,:) = ClusteringMeasure(labels, label_rmkkm);
        obj_final = [obj_final; obj];%#ok<AGROW>
        kw_aio{iRepeat} = kw;
        t_end = clock;
        disp(['RMKKM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
        disp(['RMKKM exe time: ', num2str(etime(t_end, t_start))]);
    end
    if size(rmkkm_result, 1) > 1
%         [~, minIdx] = min(obj_final);
%         rmkkm_result_obj = rmkkm_result(minIdx,:);
%         kw_obj = kw_aio{iRepeat};
        rmkkm_result_mean = mean(rmkkm_result);
    else
        rmkkm_result_mean = rmkkm_result;
    end
    save(result_file, 'rmkkm_result', 'rmkkm_result_mean', 'kernel_list', 'kw_aio');
    disp('RMKKM on multi kernel done');
end