function net = cnnFeedForward(net,x)

% Tinh toan sai so, dau ra cua mang
n = numel(net.layer);
net.layer{1}.a{1} = x; % du lieu dau vao
input_map = 1;

for la = 2: n % layer
    if strcmp(net.layer{la}.type, 'conv')
    for j = 1: net.layer{la}.outputmaps % tung noron ra se bang kernel  *& + voi tung phan no quet
        tem = zeros(size(net.layer{la - 1}.a{1},1) - net.layer{la}.kernelsize +1, ...
            size(net.layer{la - 1}.a{1},1) - net.layer{la}.kernelsize +1,size(net.layer{la - 1}.a{1},3)); % khoi tao ma tran dau ra
        for i = 1: input_map
           tem = tem + convn(net.layer{la - 1}.a{i},net.layer{la}.k{i}{j},'valid');
        end
        net.layer{la}.a{j} = logsig(tem + net.layer{la}.b{j});
    end
    input_map = net.layer{la}.outputmaps;
    else if strcmp(net.layer{la}.type, 'pool') % giam tham so :))) 
            for j = 1: input_map
                tem =  convn(net.layer{la - 1}.a{j}, ones(net.layer{la}.scale)/(net.layer{la}.scale ^ 2),'valid');
                net.layer{la}.a{j}  =tem(1 : net.layer{la}.scale : end, 1 : net.layer{la}.scale : end, :); % gia tri
            end
        end
    end 
end
 % duoi thang ra thoi 
 tem = [];
 for i = 1:input_map
     tem_size = size(net.layer{n}.a{i});
     try
        tem = [tem; reshape(net.layer{n}.a{i},tem_size(1) *tem_size(2),tem_size(3))];
     catch % neu 1 phan tu 
        tem = [tem; reshape(net.layer{n}.a{i},tem_size(1) *tem_size(2),1)];
     end
 end
 net.o = logsig( net.fcw * tem + net.fcb);
 net.fj = tem;
end
    
    

