function demo_rmkkm(n, nRepeat)
addpath('./RMKKM/');
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

result_dir = fullfile(pwd,['result_rmkkm_' num2str(nRepeat)],[dataset,'_result']);
if ~exist(result_dir,'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks=cell(nKernel,1);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    load(fullfile(kernel_dir,iFile),'K');
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
        rmkkm_result(iRepeat,:) = ClusteringMeasure(y, label_rmkkm);
        obj_final = [obj_final; obj];%#ok<AGROW>
        kw_aio{iRepeat} = kw;
        t_end = clock;
        runtime=etime(t_end, t_start);
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
    save(result_file, 'rmkkm_result', 'rmkkm_result_mean', 'kernel_list', 'kw_aio','runtime');
    disp('RMKKM on multi kernel done');
end