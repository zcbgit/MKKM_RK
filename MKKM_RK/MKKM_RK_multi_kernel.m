function MKKM_RK_multi_kernel(dataset, kernel_type, nRepeat)
data_dir = fullfile(pwd, '..', 'dataset');
kernel_dir = fullfile(pwd, '..', ['kernels' dataset, '_kernel']);
file_list = dir(kernel_dir);

kernel_list = {};
iKernel = 0;
for iFile = 1:length(file_list)
    sName = file_list(iFile).name;
    if (~strcmp(sName, '.') && ~strcmp(sName, '..'))
        if ~isempty(kernel_type)
            for iType = 1:length(kernel_type)
                if ~isempty(strfind(sName, kernel_type{iType}))
                    iKernel = iKernel + 1;
                    kernel_list{iKernel} = sName; %#ok<AGROW>
                end
            end
        else
            iKernel = iKernel + 1;
            kernel_list{iKernel} = sName; %#ok<AGROW>
        end
    end
end

load(fullfile(data_dir, dataset), 'y');
if ~exist('y', 'var')
    error(['y is not found in ', dataset]);
end
nClass = length(unique(y));

result_dir = fullfile(pwd, [dataset, '_res']);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

nKernel = length(kernel_list);
Ks = zeros(length(y), length(y), nKernel);

for iKernel = 1:length(kernel_list)
    iFile = kernel_list{iKernel};
    
    clear K;
    
    load(fullfile(kernel_dir, iFile), 'K');
    Ks(:,:,iKernel) = K;
end
clear K;

lamds_1=(0.1:0.1:1);
lamds_2=(0.1:0.1:1);
obj_final = [];
kw_aio = cell(nRepeat, 1);
disp(['Total number of Kernels: ', num2str(length(kernel_list)) '!']);

disp(['MKKM_RM on multi kernel begin ...']);
mkkm_rm_result_file = fullfile(result_dir, [dataset, '_result_mkkm_rm.mat']);
if exist(mkkm_rm_result_file, 'file')
    load(mkkm_rm_result_file, 'result_mkkm_rm_aio');
else
    rng('default');
    for i = 1:length(lamds_1)
        for j=1:length(lamds_2)
            lamd_1 = lamds_1(i);
            lamd_2 = lamds_2(j);
            result = zeros(nRepeat,3);
            for iRepeat = 1:nRepeat
                t_start = clock;
                disp(['MKKM_RM ',  num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations begin ...']);
                Y = rand(size(Ks,3));
                [~, idx] = max(Y,[],2);
                Y = zeros(size(Y));
                Y(sub2ind(size(Y),[1:size(Y,1)]',idx)) = 1;
                [label_mkkm_rm, kw, obj] = MKKM_RK(Y,nClass, 1e-5, Ks, lamd_1, lamd_2);
                result(iRepeat,:)=ClusteringMeasure(y, label_mkkm_rm);
                obj_final = [obj_final; obj];
                kw_aio{iRepeat} = kw;
                t_end = clock;
                disp(['MKKM_RM ', num2str(iRepeat), ' of ' num2str(nRepeat), ' iterations done.']);
                disp(['MKKM_RM exe time: ', num2str(etime(t_end, t_start))]);
            end
            if size(result, 1) > 1
                [~, minIdx] = min(obj_final);
                mkkm_rm_result_obj = result(minIdx,:);
                kw_obj = kw_aio{iRepeat};
                result_mean = mean(result);
            else
                result_mean = result;
            end
            save(mkkm_rm_result_file, 'result_mean');
            disp(['MKKM_RM on multi kernel done']);
            
            result_mkkm_rm_aio = result_mean;
            
            clear Ks K mkkm_rm_result_mean mkkm_rm_result_obj;
            suffix = [num2str(lamd_1) '_' num2str(lamd_2)];
            save(fullfile(result_dir, [dataset, '_result_mkkm_rm_multi_kernel_' suffix '.mat']), ...
                'result_mkkm_rm_aio', 'kernel_list');
        end
    end
end

end