[x1, Fs1] = audioread('mixed1.wav');
[x2, Fs2] = audioread('mixed2.wav');
xx = [x1, x2]';
yy = sqrtm(inv(cov(xx')))*(xx-repmat(mean(xx,2),1,size(xx,2)));
[W,s,v] = svd((repmat(sum(yy.*yy,1),size(yy,1),1).*yy)*yy');
a = W*xx;
audiowrite('refined1.wav', a(1,:), Fs1);
audiowrite('refined2.wav', a(2,:), Fs1);