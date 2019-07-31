clear all;clc;

% load data
mainFolder1 = dir('Test 1');
mainFolder1 = mainFolder1(4:end);

% create response
y = ones(1,30);
Y = [y 2*y 3*y]; 
y1 = Y; 

% Test 1
% load test 1 data
y_1 = zeros(240000,1);
Fs_1 = [];
n = 240000;
for i=1:length(mainFolder1)
    name = mainFolder1(i).name;
    a = ['Test 1','/',name];
    subfolder=dir(a);
    subfolder = subfolder(3:end);
    for j = 1:length(subfolder)
        [y,Fs] = audioread(subfolder(j).name);
        y = mean(y,2);
        %y = sum(y,2);
        if length(y(:,1))>n
            y = y(1:n,:);
            %Fs = Fs(1:n,:);
        else
            n = length(y(:,1));
            y_1 = y_1(1:n,:);
            %Fs_1 = Fs_1(1:n,:);
        end
        y_1 = [y_1 y];
        Fs_1 = [Fs_1 Fs];
    end
end
y_1 = y_1(:,2:end);
test1 = y_1;
%
% Separate into training and testing set
ind = randi([1 90],1,20);
test1_test = test1(:,ind);
y1_test = Y(:,ind);
test1(:,ind) = [];
y1(:,ind) = [];
test1_train = test1;
y1_train = y1;

%
spec_train = [];
for i = 1:length(test1_train(1,:))
    ft = fft(test1_train(:,i));
    spec = abs(fftshift(ft));
    spec_train = [spec_train spec];
end

spec_test = [];
for i = 1:length(test1_test(1,:))
    ft = fft(test1_test(:,i));
    spec = abs(fftshift(ft));
    spec_test = [spec_test spec];
end

[a,b]=size(spec_train); % compute data size
ab=mean(spec_train,2); % compute mean for each row
spec_train=spec_train-repmat(ab,1,b); % subtract mean

[c,d]=size(spec_test); % compute data size
cd=mean(spec_test,2); % compute mean for each row
spec_test=spec_test-repmat(cd,1,d); % subtract mean

%
[U1,S1,V1] = svd(spec_train','econ');
plot(diag(S1)/sum(diag(S1)),'ro','LineWidth',[2])
xlabel('Song clips spectrogram')
ylabel('Singular values')
title('Singular value spectrum for Test 1')
print(gcf,'-dpng','test1_singular_value_spectrum.png');

% KNN
knn.mod = fitcknn(V1',y1_train','NumNeighbors',5);
label = predict(knn.mod,test1_test');
right = 0;
for i = 1:length(label)
    if label(i) == y1_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 1 using KNN is ', num2str(accuracy)];
disp(A)

%% SVM
svm.mod = fitcecoc(V1',y1_train');
label = predict(svm.mod,test1_test');
right = 0;
for i = 1:length(label)
    if label(i) == y1_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 1 using SVM is ', num2str(accuracy)];
disp(A)

%% LDA
lda.mod = fitcdiscr(V1',y1_train');
label = predict(lda.mod,test1_test');
right = 0;
for i = 1:length(label)
    if label(i) == y1_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 1 using LDA is ', num2str(accuracy)];
disp(A)





%% SVM
svm.mod1 = fitcsvm(V1',y1_train','ClassNames',{'not 1','1'});
svm.mod2 = fitcsvm(V1',y1_train','ClassNames',{'not 2','2'});
svm.mod3 = fitcsvm(V1',y1_train','ClassNames',{'not 3','3'});
label1 = predict(svm.mod1,spec_test');
label2 = predict(svm.mod2,spec_test');
label3 = predict(svm.mod3,spec_test');

%% Naive Bayes
nb.mod = fitcnb(V1',y1_train');
label = predict(nb.mod,test1_test');
right = 0;
for i = 1:length(label)
    if label(i) == y1_test(i)
        right = right + 1;
    end
end
accuracy = right/10;
A = ['Accuracy for test 1 using Naive Bayes is ', num2str(accuracy)];
disp(A)
%%
% get the average artist vector
test1_ave1 = sum(artist1,2)/length(artist1(1,:)); % average music for each artist
test1_ave2 = sum(artist2,2)/length(artist2(1,:));
test1_ave3 = sum(artist3,2)/length(artist3(1,:));

% SVD
[U1,S1,V1] = svd(test1_train','econ');
figure(1)
plot(diag(S1)/sum(diag(S1)),'ro','LineWidth',[2])
xlabel('Song clips')
ylabel('Singular values')
title('Singular value spectrum for Test 1')
print(gcf,'-dpng','test1_singular_value_spectrum.png');
% rank = 10

proj_X1_train = V1'*test1_train;
knn.mod = fitcknn(proj_X1_train',y1_train,'NumNeighbors',5);
proj_X1_test = V1'*test1_test;
label = predict(knn.mod,proj_X1_test');




%%
% project average artist on the new basis
test1_proj1 = V1'*test1_ave1;
test1_proj2 = V1'*test1_ave2;
test1_proj3 = V1'*test1_ave3;

% testing on test set
proj = V1'*test1_test;
accurate = 0;
R = [];
for i = 1:length(y1_test)
    MSE = [mean(abs(proj(:,i)-test1_proj1))
        mean(abs(proj(:,i)-test1_proj2))
        mean(abs(proj(:,i)-test1_proj3))];
    [M,I] = min(MSE);
    
    if I == 1
        r = 1;
    elseif I == 2
        r = 2;
    else
        r = 3;
    end
    R = [R r];
    
    if y1_test(i) == r
        accurate = accurate + 1;
    end
end
accuracy = sum(accurate)/10;
A = ['Accuracy for test 1 is ', num2str(accuracy)];
disp(A)

