clear, clc, clf
addpath(genpath('yaml_matlab')); 
config = ReadYaml('config.yaml');
log_folder = [config.dropbox,'logs/'];
% log_train_name = 'FCN_train_log_Dec08_0215.txt';
% log_test_name = 'FCN_test_log_Dec08_0215.txt';
% log_train_name = 'DilatedNet_train_log_Dec08_0230.txt';
% log_test_name = 'DilatedNet_test_log_Dec08_0230.txt';
log_train_name = 'DilatedNet_train_log_Dec08_1115.txt';
log_test_name = 'DilatedNet_test_log_Dec08_1115.txt';
log_train_file = [log_folder,log_train_name];
T_train = readtable(log_train_file,'Delimiter',',','HeaderLines',0,'ReadVariableNames',true);
vars = T_train.Properties.VariableNames;

N = size(vars,2)-2;
for i = 1:N
    h(i) = figure(i);
end
set(h,'WindowStyle','Docked');

xname = vars{1};
xdata_train = T_train.(xname);
k = max(round(length(xdata_train)/20),1);
% k = 1
for i = 1:N
    yname = vars{i+2};
    figure(i)
    ydata_train = movmean(T_train.(yname),k);
    plot(xdata_train,ydata_train)
    title(yname)
    xlabel(xname)
    ylabel(yname)
end

validation_too = true;

if validation_too
    test_interval = 10;
    log_test_file = [log_folder,log_test_name];
    T_test= readtable(log_test_file,'Delimiter',',','HeaderLines',0,'ReadVariableNames',true);
    xdata_test = test_interval*T_test.(xname);
    for i = 1:N
        yname = vars{i+2};
        ydata_test = movmean(T_test.(yname),max(round(k/test_interval),1));
        figure(i)
        hold on
        plot(xdata_test,ydata_test)
        hold off
        legend('training','validation')
    end
end

