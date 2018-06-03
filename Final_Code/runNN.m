function [y] = runNN(net,x)
% Dau vao : Net + dau vao
% Dau ra: Nhan du doan
    tem = cnnFeedForward(net,x);
    [~,duDoan] = max(tem.o); 
    y = duDoan - 1;
     try
        for i = 1 :  size(x,3)
            figure

            imshow(x(:,:,i)); % hien thi anh
            t = ['Predict: ', num2str(y(i))];
            title(t);
        end
     catch % neu 1 phan tu 
        figure
        imshow(x); % hien thi anh
     end

end