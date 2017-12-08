clear, clc, clf
log_folder = '/home/amado/dropbox/Local/Fall17/Computer_Vision/Scene_Segmentation/logs/';
% log_name = 'DilatedNet_log.txt';
log_name = 'FCN_log.txt';
log_file = [log_folder,log_name];
T = readtable(log_file,'Delimiter',',','HeaderLines',0,'ReadVariableNames',true);
variables = T.Properties.VariableNames;

xname = variables{1};
xdata = T.(xname);
k = 100;
for i = 3:size(variables,2)
    yname = variables{i};
    figure('Name',yname)
    ydata = movmean(T.(yname),k);
    plot(xdata,ydata)
    xlabel(xname)
    ylabel(yname)
end