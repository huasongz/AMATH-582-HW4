clear all;clc;

% load files
mainFolder1 = dir('CroppedYale');
mainFolder1 = mainFolder1(4:41);
mainFolder2 = dir('yalefaces_uncropped');
mainFolder2 = mainFolder2(4);

%% Cropped
ave_face = [];
CROP = [];
for i=1:length(mainFolder1)
    name = mainFolder1(i).name;
    a = ['CroppedYale','/',name];
    subfolder=dir(a);
    subfolder = subfolder(3:end);
    Cropped = [];
    for j = 1:length(subfolder)
        data = imread(subfolder(j).name);
        data = reshape(data,192*168,1);
        Cropped = [Cropped data];
        CROP = [CROP data]; 
    end
    ave_face = [ave_face sum(Cropped,2)/length(Cropped(1,:))];
end
% CROP: all the images, each column represents an image
% ave_face: each column is the average face for each person

% SVD
CROP = double(CROP);
[U,S,V] = svd(CROP,'econ');

% Plot
figure(1)
plot(diag(S)/sum(diag(S)),'ro','LineWidth',[2])
ylabel('Singular Values')
xlabel('Average Faces Images')
title('Singular Value Spectrum on cropped images')
print(gcf,'-dpng','crop_singular_value_spectrum.png');
% The plot shows it has rank 4
%%
reconstruct = U*S(:,1:50)*V(:,1:50)';

img = [2,50,100];
figure(2)
for i = 1:length(img)
    subplot(3,2,2*i-1)
    imshow(uint8(reshape(CROP(:,img(i)),192,168)));
    title('Original Image')
    subplot(3,2,2*i)
    imshow(uint8(reshape(reconstruct(:,img(i)),192,168)));
    title('Reconstructed Image')
end
print(gcf,'-dpng','Reconstruct_cropped_image.png');


test = imread('yaleB33_P00A+010E+00.pgm');
figure(3)
subplot(2,3,1)
imshow(test)
title('Test Image')
test = double(reshape(test,192*168,1));
% rank 4
rank = [4,10,50,100,200];

for i = 1:length(rank)
    U_new = U(:,1:rank(i));
    recon = U_new*U_new'*test;
    recon = reshape(recon,192,168);
    subplot(2,3,i+1)
    imshow(uint8(recon))
    t = ['r = ',num2str(rank(i))];
    title(t)
end
print(gcf,'-dpng','Reconstruct_cropped_image1.png');

%% Original
%ave_face_original = [];
ORIGINAL = [];
for i=1:length(mainFolder2)
    name = mainFolder2(i).name;
    a = ['yalefaces_uncropped','/',name];
    subfolder=dir(a);
    subfolder = subfolder(3:end);
    original = [];
    for j = 1:length(subfolder)
        data = imread(subfolder(j).name);
        data = reshape(data,243*320,1);
        original = [original data];
        ORIGINAL = [ORIGINAL data]; 
    end
    %ave_face_original = [ave_face_original sum(original,2)/length(original(1,:))];
end
% CROP: all the images, each column represents an image
% ave_face: each column is the average face for each person

% SVD
ORIGINAL = double(ORIGINAL);
[U2,S2,V2] = svd(ORIGINAL,'econ');
%[U3,S3,V3] = svd(ave_face_original,'econ');

% Plot
figure(4)
plot(diag(S2)/sum(diag(S2)),'ro','LineWidth',[2])
ylabel('Singular Values')
xlabel('Images')
print(gcf,'-dpng','original_singular_value_spectrum.png');
% The plot shows it has rank 2

% Test
[test,B]= imread('subject03.normal');
%figure(4)
%subplot(2,3,1)
%imshow(test)
%title('Test Image')
%test = double(reshape(test,243*320,1));
% rank 4
reconstruct = U2*S2(:,1:50)*V2(:,1:50)';

img = [2,50,100];
figure(5)
for i = 1:length(img)
    subplot(3,2,2*i-1)
    imshow(uint8(reshape(ORIGINAL(:,img(i)),243,320)));
    title('Original Image')
    subplot(3,2,2*i)
    imshow(uint8(reshape(reconstruct(:,img(i)),243,320)));
    title('Reconstructed Image')
end
print(gcf,'-dpng','Reconstruct_original_image.png');