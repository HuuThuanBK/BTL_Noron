function [err_rate,err_band] = cnnTest(net, x, y)
% err_band: loi tu 0 -> 9 :

    err_band = zeros(1,10);
    err = 0; 
    tem = cnnFeedForward(net,x);
    [~,duDoan] = max(tem.o); 
    [~,thuc] = max(y);% vd: thuc 1 : la so 0
    
    for i = 1: size(y,2)
        if duDoan(i) ~= thuc(i)
           err_band(thuc(i))  = err_band(thuc(i)) + 1;
           err = err+1;
        end
    end
    
    for i = 1 :10
        err_band(i) = round(err_band(i) / size(find(thuc == i),2),5);
    end
    err_rate = err/size(y,2); 
end