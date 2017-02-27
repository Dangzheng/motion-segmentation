clear;clc;close all;
u = fopen('u.txt','r');
v = fopen('v.txt','r');
img = imread('1.png');
img_hat = imread('3.png');
img = rgb2gray(img);
img_hat = rgb2gray(img_hat);
img = im2double(img);
img_hat = im2double(img_hat);
[m,n] = size(img);
[U,~] = fscanf(u,'%f %f',[n,m]);
gt = flow_read('000000_10.png');
u_gt = gt(:,:,1);
v_gt = gt(:,:,2);
mask = gt(:,:,3);
[V,~] = fscanf(v,'%f %f',[n,m]);
close all;
u = U';v = V';
% u = round(U');v = round(V');
u(mask>0) = u_gt(mask>0);
v(mask>0) = v_gt(mask>0);
[x,y] = meshgrid(1:1:n,1:1:m);
x_hat = x+u;
y_hat = y+v;

% img_warp = griddata(x,y,img_hat,x_hat,y_hat);
img_warp = griddata(x_hat,y_hat,img,x,y);
imshow(img_warp,[]);

u_sys = fopen('u_sintel_back.txt','w');
v_sys = fopen('v_sintel_back.txt','w');

for row = 1:1:m
    for col = 1:1:n-1
        data = u_gt(row,col);
        fprintf(u_sys ,'%f ', data);
        data = v_gt(row,col);
        fprintf(v_sys,'%f ',data);
    end
    col = n;
    data = u_gt(row,col);
    fprintf(u_sys,'%f\n', data);
    data = v_gt(row,col);
    fprintf(v_sys,'%f\n', data);
end
close all;