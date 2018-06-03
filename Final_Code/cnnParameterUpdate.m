function net = cnnParameterUpdate(net, opts)
% cap nhat trong so
for la = 2 : numel(net.layer)
    if strcmp(net.layer{la}.type, 'conv')
        for j = 1: numel(net.layer{la}.a)
            for i = 1: numel(net.layer{la - 1 }.a)
                net.layer{la}.k{i}{j} = net.layer{la}.k{i}{j} - opts.alpha * net.layer{la}.dentaK{i}{j};
            end
            net.layer{la}.b{j} = net.layer{la}.b{j} - opts.alpha * net.layer{la}.dentaB{j};
        end
    end
end
net.fcw = net.fcw - opts.alpha * net.dfcw;
net.fcb = net.fcb - opts.alpha * net.dfcb;
end
                
    
