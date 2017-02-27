close; clear; clc;

flow_fore(:,:,1) = load('full data/u_epic_sintel.txt');
flow_fore(:,:,2) = load('full data/v_epic_sintel.txt');
flow_back(:,:,1) = load('full data/u_epic_sintel_back.txt');
flow_back(:,:,2) = load('full data/v_epic_sintel_back.txt');
img = im2double(imread('full data/frame_0020.png'));

fact = 2;

[m,n,~] = size(img);
height = m/fact;
width = n/fact;
m = height;
n = width;

% origIntr = intr;
% intr(1,:) = intr(1,:)/fact ; intr(3,3) = 1;
% intr(2,:) = intr(2,:)/fact ; intr(3,3) = 1;

flow_fore = resizeFlow(flow_fore, [m,n]);
flow_back = resizeFlow(flow_back, [m,n]);
img = imresize(img, [m, n]);
imwrite(img,'frame_0020.png');
flow2txt(flow_fore,'epic_sintel');
flow2txt(flow_back,'epic_sintel_back');


