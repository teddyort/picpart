clear, clc
addpath(genpath('yaml_matlab')); 
config = ReadYaml('config.yaml');
data_folder = [config.dropbox,'data/ADEChallengeData2016/'];
object_info = [data_folder,'objectInfo150.csv'];
T = readtable(object_info,'Delimiter',',','HeaderLines',0,...
    'ReadVariableNames',true);
freq = T.Train/sum(T.Train);
weights = median(freq)./freq;

F = fopen([data_folder,'class_weights.txt'],'w');

for i = 1:length(weights)
    fprintf(F,'    class_weighting: %5.3f\t# %d\n',weights(i),i);
end

fclose(F);