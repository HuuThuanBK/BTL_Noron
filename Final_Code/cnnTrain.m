function net = cnnTrain(net, x, y, opts, err_pause)
% err_pause: min den 1 gia tri nhat dinh thi dung
%dk : min lien tuc check_err lan thi moi ok
check_err = 6;


numbatches =  round(size(x, 3)/opts.batchsize); % 1 epochs thi duoc numbatches

count = 0;
last = [];
tic;
for i = 1 : opts.numepochs
	fprintf('Epoch: %d/%d \n', i, opts.numepochs);
	tem_count = randperm(size(x, 3));% lay ngau nhien 0 -> size x: 60000
    if i > 40
        opts.alpha = opts.alpha2;
    else if i> 80
         opts.alpha = opts.alpha3;
        else if i > 110
            opts.alpha = opts.alpha4;
            else if i > 130
                  opts.alpha = opts.alpha5;
                end
            end
        end
    end
    
	for j = 1: numbatches
		x_mini = x(:,:,tem_count((j-1)*opts.batchsize + 1 : j*opts.batchsize));
		y_mini = y(:, tem_count((j-1)*opts.batchsize + 1 : j*opts.batchsize));
		
		net = cnnFeedForward(net,x_mini);

		net = cnnBackPropagation(net,y_mini);

		
		net = cnnParameterUpdate(net, opts);

        t = ['Epoch: ', num2str(i), '--Numbatches(of Epoch): ' num2str(j), '--Error: ' ,num2str(round(net.L, 4)), '--Time: ',num2str(round(toc,2)),'s'];
        if (count == 0) || (count == 1)
            last1 = [count net.L];
            
        else

            title(t);
            plot([last1(1) count] , [last1(2) net.L],'-*r');
            hold on 
            xlabel('Numbatches')
            ylabel('Error')
            last1 = [count net.L];
          
        end
%         fprintf('Err: %f \n' , net.L);
        pause(0.000001);
		count = count +1;
    end
    
end
hold off
end

