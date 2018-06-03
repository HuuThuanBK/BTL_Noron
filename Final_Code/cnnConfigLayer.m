function cnnConfig = cnnConfigLayer()
cnnConfig.layer{1}.type = 'input';
cnnConfig.layer{1}.dimension = [28 28 1];

cnnConfig.layer{2}.type = 'conv';
cnnConfig.layer{2}.outputmaps = 6; % so noron lop ay
cnnConfig.layer{2}.kernelsize = 5;

% cnnConfig.layer{2}.outputmaps = 8; % so noron lop ay
% cnnConfig.layer{2}.kernelsize = 3;

cnnConfig.layer{3}.type = 'pool';
cnnConfig.layer{3}.scale = 2;

cnnConfig.layer{4}.type = 'conv';
cnnConfig.layer{4}.outputmaps = 12; % so noron lop ay
cnnConfig.layer{4}.kernelsize = 5;
% cnnConfig.layer{4}.outputmaps = 16; % so noron lop ay
% cnnConfig.layer{4}.kernelsize = 4;

cnnConfig.layer{5}.type = 'pool';
cnnConfig.layer{5}.scale = 2;
end


