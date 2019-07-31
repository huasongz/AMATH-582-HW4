clear all;clc;

mainFolder3 = dir('Test 3');
mainFolder3 = mainFolder3(4:end);

y = ones(1,30);
Y = [y 2*y 3*y]; 
y3 = Y;

% Test 2
% Load Test 2 data
y_3 = zeros(240000,1);
Fs_3 = [];
n = 240000;
for i=1:length(mainFolder3)
    name = mainFolder3(i).name;
    a = ['Test 3','/',name];
    subfolder=dir(a);
    subfolder = subfolder(3:end);
    for j = 1:length(subfolder)
        [y,Fs] = audioread(subfolder(j).name);
        y = mean(y,2);
        if length(y(:,1))>n
            y = y(1:n,:);
            %Fs = Fs(1:n,:);
        else
            n = length(y(:,1));
            y_3 = y_3(1:n,:);
            %Fs_1 = Fs_1(1:n,:);
        end
        y_3 = [y_3 y];
        Fs_3 = [Fs_3 Fs];
    end
end
y_3 = y_3(:,2:end);
test3 = [];
for j = 1:2:length(y_3(1,:))-1
    b = reshape(y_3(:,j:j+1),length(y_3(:,1))*2,1);
    test3 = [test3 b];
end
%y_3 = y_3(:,2:end);
test3 = y_3;

% Separate into training and testing set
ind = randi([1 90],1,20);
test3_test = test3(:,ind);
y3_test = y3(:,ind);
test3(:,ind) = [];
y3(:,ind) = [];
test3_train = test3;
y3_train = y3;

% Spectrogram
spec_train = [];
for i = 1:length(test3_train(1,:))
    ft = fft(test3_train(:,i));
    spec = abs(fftshift(ft));
    spec_train = [spec_train spec];
end

spec_test = [];
for i = 1:length(test3_test(1,:))
    ft = fft(test3_test(:,i));
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
[U3,S3,V3] = svd(spec_train','econ');
figure(3)
plot(diag(S3)/sum(diag(S3)),'ro','LineWidth',[2])
xlabel('Song clips spectrogram')
ylabel('Singular values')
title('Singular value spectrum for Test 3')
print(gcf,'-dpng','test3_singular_value_spectrum.png');
% rank = 10

% KNN
knn.mod = fitcknn(V3',y3_train','NumNeighbors',5);
label = predict(knn.mod,test3_test');
right = 0;
for i = 1:length(label)
    if label(i) == y3_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 3 using KNN is ', num2str(accuracy)];
disp(A)

% SVM
svm.mod = fitcecoc(V3',y3_train');
label = predict(svm.mod,test3_test');
right = 0;
for i = 1:length(label)
    if label(i) == y3_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 3 using SVM is ', num2str(accuracy)];
disp(A)