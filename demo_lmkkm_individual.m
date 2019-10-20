function demo_lmkkm_individual(m)
addpath('./LMKKM/');
addpath('./lib');
datasets={'binaryalphadigs_1404n_320d_36c','COIL20_1440n_1024d_20c'};
dims=[length(datasets),10];
index=indexTransRowwise(m,dims);
n=index(1);
iRepeat=index(2);
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

result_dir = fullfile(pwd,'result_lmkkm_individual',[dataset,'_result']);
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