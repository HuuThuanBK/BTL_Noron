function net = cnnInit(net, x, y) % khoi tao cac tham so mang

% input: Mo Hinh Mang, du lieu hoc
% output: Khoi tao cac thong so mo hinh mang
% ex: C1: k = 5x5, C2 = 5x5, w = 10*192

input_map = 1; % so noron cua 1 lop-Lop vao
size_map = size((x(:, :, 1)));  % kich thuoc axa du lieu vao

num_layer = numel(net.layer);

for l = 2: num_layer
    if strcmp(net.layer{l}.type, 'conv')
       size_map  = size_map - net.layer{l}.kernelsize +1;
      tem_in =  input_map* net.layer{l}.kernelsize^2;
      tem_out =  net.layer{l}.outputmaps* net.layer{l}.kernelsize^2;
      
      for j=1: net.layer{l}.outputmaps
             for i = 1: input_map
              net.layer{l}.k{i}{j} = (rand(net.layer{l}.kernelsize) -0.5)*2* sqrt(6 / (tem_in + tem_out)); % CT co san nhe :))
              net.layer{l}.b{j} = 0;
          end
      end
          input_map = net.layer{l}.outputmaps; 
    end
    if strcmp(net.layer{l}.type, 'pool')
        size_map = size_map/net.layer{l}.scale;
        
    end
end
f = prod(size_map) *  input_map; % duoi thang cac phan tu: 192x1;
out = size(y,1);
net.fcb = zeros(out, 1);
net.fcw = (rand ( out,f) -0.5)*2*sqrt(6 / (out + f));
end