function A = cnnUpsampling(B,scale)
% dau vao: ma tran, scale 
% lap hang, cot
% dau ra: ma tran 
sizeB = size(B);
A = [];
for hang = 1: sizeB(1)
    tem = [];
    for cot = 1 : sizeB(2)
        for i = 1: scale
            tem = [tem B(hang,cot,:)];
        end
    end
    for i = 1: scale
        A = [A;tem];
    end
end
end
