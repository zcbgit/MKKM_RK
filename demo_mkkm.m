function demo_mkkm(n, nRepeat)
addpath('./MKKM/');
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

result_dir = fullfile(pwd,['result_mkkm_' num2str(nRepeat)],[dataset,'_result']);
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

kw_aio = cell(nRepeat,1);
disp(['Total number of Kernels: ', num2str(length(kernel_list))]);

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
    mkkm_result = [mkkm_result; ClusteringMeasure(y, label_mkkm)];%#ok<AGROW>
    obj_final = [obj_final; obj];%#ok<AGROW>
    kw_aio{iRepeat} = kw;
    t_end = clock;
    runtime=etime(t_end, t_start);
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
save(result_file, 'mkkm_result', 'mkkm_result_mean', 'kernel_list', 'kw_aio','runtime');
disp('MKKM on multi kernel done');