%Backpropagation
function net = cnnBackPropagation(net,y)
% dau vao: noron, nhan dung
% dau ra: cac he so denta de cap nhat
% quy uoc: ct - cong thuc

n = numel(net.layer);

% sai e: output - trueLabe
net.e = net.o - y;
%tinh loi: Loss Function
net.L = 1/2*sum((net.e(:)).^2) /size(net.e,2); % net.e,2 : so mau

%% Tinh cho truong hop 1 batch size = x, ( giong trong cong thuc trong bao)
 net.fcb_tem = (net.e).*net.o.*(1-net.o); % ct 23,24
 net.dentaF = net.fcw'* net.fcb_tem; % ct 30
 
 tem_size = size(net.layer{n}.a{1}); %  size 1 cai noron
 
 %% tinh cac trong so denta W(dfcw) va denta b(dfcb)
net.dfcw = (net.fcb_tem * (net.fj)') ./ tem_size(3); % ct 19, 
% ------------10 x batch size .... 192 x batch size
net.dfcb = mean(net.fcb_tem,2); % lam tron theo hang
 
 % doi nguoc lai thanh ma tram 4X4(12) tu 192X1X( batch size)
  count = 1; % de dem thoi :))
  
 for j = 1: numel(net.layer{n}.a) % trong truong hop nay la 12
	net.layer{n}.dentaS{j} = [];
	tem = [];
	for i = 1: (tem_size(1) * tem_size(2))
		tem = [tem ; net.dentaF(count, 1:tem_size(3))];
		count = count+1;
    end
%     size(tem)     
%     tem(1: size(tem,1),:),
	net.layer{n}.dentaS{j} = reshape(tem(1: size(tem,1),:),tem_size(1),tem_size(2),tem_size(3));
end

for la = (n -1 ) : -1 : 1 
	if strcmp(net.layer{la}.type, 'conv')
		for j = 1: numel(net.layer{la}.a) % duyet het ouput
			% cong thuc 32: su dung lay phan nguyen tren: tuc la se bi nap lai 4x4 -> 8x8
			net.layer{la}.dentaC{j} = cnnUpsampling(net.layer{la + 1}.dentaS{j}, net.layer{la +1}.scale) ./ (net.layer{la +1}.scale^2) ;
			net.layer{la}.dentaC_teta{j} = net.layer{la}.dentaC{j} .* net.layer{la}.a{j} .* ( 1 - net.layer{la}.a{j}); % ct 37

		end
	else if strcmp(net.layer{la}.type, 'pool')
	
		% Tinh toan denta S cho phan sau
		for i = 1 : numel(net.layer{la}.a)
			tem = zeros(size(net.layer{la}.a{1}));
			for j = 1: numel(net.layer{la + 1}.a)
 				tem = tem + convn(net.layer{la + 1}.dentaC_teta{j},rot90(net.layer{la + 1}.k{i}{j},2)); % ct 51
                % rot 90 no khong thanh phan thu 3(:,:,x) 
			end
			net.layer{la}.dentaS{i} = tem;
		end
			
	
		% Tinh toan denta k va b
		for j = 1: numel(net.layer{la + 1}.a)
			for i = 1 : numel(net.layer{la}.a)
				net.layer{la+1}.dentaK{i}{j} = convn(flipPlus(net.layer{la}.a{i}), net.layer{la+1}.dentaC_teta{j}, 'valid') ./ tem_size(3); % rot90(A,2), which rotates by 180 degrees.
			end
			net.layer{la+1}.dentaB{j} = sum(net.layer{la+1}.dentaC_teta{j}(:)) ./ tem_size(3) ; % chia cho tong so mau
		end
        
        else
        for j = 1: numel(net.layer{la + 1}.a)
			for i = 1 : numel(net.layer{la}.a)
%  				net.layer{la+1}.dentaK{i}{j} = convn(rot90(net.layer{la}.a{i},2), net.layer{la+1}.dentaC_teta{j}, 'valid') ./ tem_size(3); % rot90(A,2), which rotates by 180 degrees.
                net.layer{la+1}.dentaK{i}{j} = convn(flipPlus(net.layer{la}.a{i}), net.layer{la+1}.dentaC_teta{j}, 'valid') ./ tem_size(3);
            end
			net.layer{la+1}.dentaB{j} = sum(net.layer{la+1}.dentaC_teta{j}(:)) ./ tem_size(3) ; % chia cho tong so mau
        end
	end
	end
end
	
%%
end

















