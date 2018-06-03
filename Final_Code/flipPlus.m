function A = flipPlus(A)
% thay doi luon ca 3 chieu rot90(X,2) ... 
for i = 1: ndims(A) % so chieu, thuong 3 chieu
    A = flip(A,i);
end
end