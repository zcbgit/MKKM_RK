function demo_mkkm_rk_sfn(n, nRepeat)
addpath('./MKKM_RK/');
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

lambdas=[0,2.^(-20:2)];
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

result_dir = fullfile(pwd,['result_mkkm_rk_sfn' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks = zeros(length(y),length(y),nKernel);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    load(fullfile(kernel_dir,iFile),'K');
    Ks(:,:,iKernel) = K;
end

disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

disp('MKKM with Representative Kernels begin ...');
for i = 1:length(lambdas)
    lambda=lambdas(i);
    mkkm_rk_result = zeros(nRepeat,3);
    obj_final = cell(nRepeat,1);
    kw_aio = cell(nRepeat,1);
    Y_final = cell(nRepeat,1);
    suffix = num2str(lambda);
    result_file = fullfile(result_dir,[dataset,'_mkkm_rk_' suffix '.mat']);
    % rng('default');
    for iRepeat = 1:nRepeat
        t_start = clock;
        disp(['MKKM_RK ',num2str(iRepeat),' of ' num2str(nRepeat), ' iteration begin ...']);
        Y = rand(size(Ks,3));
        [~, idx] = max(Y,[],2);
        Y = zeros(size(Y));
        Y(sub2ind(size(Y),[1:size(Y,1)]',idx)) = 1;
        [label,kw,obj,Y] = MKKM_RK(Y,nClass,1e-5,Ks,'sfn',lambda);
        mkkm_rk_result(iRepeat,:) = ClusteringMeasure(y,label);
        obj_final{iRepeat} = obj;
        kw_aio{iRepeat} = kw;
        Y_final{iRepeat} = Y;
        t_end = clock;
        runtime=etime(t_end, t_start);
        disp(['MKKM_RK ',num2str(iRepeat),' of ' num2str(nRepeat),' iterations done.']);
        disp(['MKKM_RK exe time: ',num2str(etime(t_end, t_start))]);
    end
    if size(mkkm_rk_result,1) > 1
        mkkm_rk_result_mean = mean(mkkm_rk_result);
    else
        mkkm_rk_result_mean = mkkm_rk_result;
    end
    save(result_file,'mkkm_rk_result','mkkm_rk_result_mean','kernel_list',...
        'kw_aio','obj_final','Y_final','runtime');
    disp('MKKM_RK on multi kernel done');
end