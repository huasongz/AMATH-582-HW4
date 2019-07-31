clear all;clc;

mainFolder2 = dir('Test 2');
mainFolder2 = mainFolder2(4:end);

y = ones(1,30);
Y = [y 2*y 3*y]; 
y2 = Y;

% Test 2
% Load Test 2 data
y_2 = zeros(240000,1);
Fs_2 = [];
n = 240000;
for i=1:length(mainFolder2)
    name = mainFolder2(i).name;
    a = ['Test 2','/',name];
    subfolder=dir(a);
    subfolder = subfolder(3:end);
    for j = 1:length(subfolder)
        [y,Fs] = audioread(subfolder(j).name);
        y = mean(y,2);
        if length(y(:,1))>n
            y = y(1:n,:);
        else
            n = length(y(:,1));
            y_2 = y_2(1:n,:);
        end
        y_2 = [y_2 y];
        Fs_2 = [Fs_2 Fs];
    end
end
y_2 = y_2(:,2:end);
test2 = y_2;

% Separate into training and testing set
ind = randi([1 90],1,10);
test2_test = test2(:,ind);
y2_test = y2(:,ind);
test2(:,ind) = [];
y2(:,ind) = [];
test2_train = test2;
y2_train = y2;

spec_train = [];
for i = 1:length(test2_train(1,:))
    ft = fft(test2_train(:,i));
    spec = abs(fftshift(ft));
    spec_train = [spec_train spec];
end

spec_test = [];
for i = 1:length(test2_test(1,:))
    ft = fft(test2_test(:,i));
    spec = abs(fftshift(ft));
    spec_test = [spec_test spec];
end

[a,b]=size(spec_train); % compute data size
ab=mean(spec_train,2); % compute mean for each row
spec_train=spec_train-repmat(ab,1,b); % subtract mean

[c,d]=size(spec_test); % compute data size
cd=mean(spec_test,2); % compute mean for each row
spec_test=spec_test-repmat(cd,1,d); % subtract mean

% SVD
[U2,S2,V2] = svd(spec_train','econ');
figure(2)
plot(diag(S2)/sum(diag(S2)),'ro','LineWidth',[2])
xlabel('Song clips spectrogram')
ylabel('Singular values')
title('Singular value spectrum for Test 2')
print(gcf,'-dpng','test2_singular_value_spectrum.png');
% rank = 10

% KNN
knn.mod = fitcknn(V2',y2_train','NumNeighbors',5);
label = predict(knn.mod,test2_test');
right = 0;
for i = 1:length(label)
    if label(i) == y2_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 2 using KNN is ', num2str(accuracy)];
disp(A)

% SVM
svm.mod = fitcecoc(V2',y2_train');
label = predict(svm.mod,test2_test');
right = 0;
for i = 1:length(label)
    if label(i) == y2_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 2 using SVM is ', num2str(accuracy)];
disp(A)

