% test 
clear;
clc;
load mnist_uint8;
train_x = double(reshape(train_x',28,28,[]))/255;
train_x = permute(train_x, [2 1 3]);
train_y = double(train_y');

test_x = double(reshape(test_x',28,28,[]))/255;
test_x = permute(test_x, [2 1 3]);
test_y = double(test_y');

net = cnnConfigLayer();
opts = cnnConfigParameters();
net = cnnInit(net, train_x, train_y);
% nettest = net;

% net = cnnFeedForward(net,train_x(:,:,1:2))
% net = cnnBackPropagation(net,train_y(:,1:2))
net = cnnTrain(net, train_x, train_y, opts, 0.1);
%%
[err, err_band] = cnnTest(net,test_x,test_y);

hold off
figure
%plot: ve do sai so voi data test
tem = [];

for i = 1:10
    tem = [tem;100 - err_band(i)*100,err_band(i)*100];

end

bar(0:9,tem,'stacked')
ylim([0 105]);
xlim([-0.5 10]);
for i = 1: 10
    str = sprintf('%.2f', 100 - err_band(i)*100);
    text(i-1 -0.3 ,50,str,'fontsize', 8);
end
legend('Correct','Error')
title(['Sai so tung truong hop voi Data Testing']);


%%
hold off
figure

[err_train, err_band_train] = cnnTest(net,train_x,train_y);

%plot: ve do sai so voi data train
tem = [];

for i = 1:10
    tem = [tem;100 - err_band_train(i)*100,err_band_train(i)*100];

end

b = bar(0:9,tem,'stacked');
ylim([0 105]);
xlim([-0.5 10]);
for i = 1: 10
    str = sprintf('%.2f', 100 - err_band_train(i)*100);
    text(i-1 -0.3 ,50,str,'fontsize', 8);
end
legend('Correct','Error')
title(['Sai so tung truong hop voi Data Training']);
%%