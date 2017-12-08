clear, clc, clf
addpath(genpath('yaml_matlab')); 
config = ReadYaml('config.yaml');
log_folder = [config.dropbox,'logs/'];
% log_train_name = 'DilatedNet_log.txt';
log_train_name = 'FCN_train_log.txt';
log_train_file = [log_folder,log_train_name];
T_train = readtable(log_train_file,'Delimiter',',','HeaderLines',0,'ReadVariableNames',true);
vars = T_train.Properties.VariableNames;

xname = vars{1};
xdata_train = T_train.(xname);
k = 400;
for i = 3:size(vars,2)
    yname = vars{i};
    figure(i-2)
    ydata_train = movmean(T_train.(yname),k);
    plot(xdata_train,ydata_train)
    title(yname)
    xlabel(xname)
    ylabel(yname)
end

validation_too = true;

if validation_too
    test_interval = 10;
    log_test_name = 'FCN_test_log.txt';
    log_test_file = [log_folder,log_test_name];
    T_test= readtable(log_test_file,'Delimiter',',','HeaderLines',0,'ReadVariableNames',true);
    xdata_test = test_interval*T_test.(xname);
    for i = 3:size(vars,2)
        yname = vars{i};
        ydata_test = movmean(T_test.(yname),round(k/test_interval));
        figure(i-2)
        hold on
        plot(xdata_test,ydata_test)
        hold off
        legend('training','validation')
    end
end
