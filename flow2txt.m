
function flow2txt(flowuv,outputname)
[m,n,~] = size(flowuv);
u = flowuv(:,:,1);
v = flowuv(:,:,2);
u_sys = fopen(['u_',outputname,'.txt'],'w');
v_sys = fopen(['v_',outputname,'.txt'],'w');

for row = 1:1:m
    for col = 1:1:n-1
        data = u(row,col);
        fprintf(u_sys ,'%f ', data);
        data = v(row,col);
        fprintf(v_sys,'%f ',data);
    end
    col = n;
    data = u(row,col);
    fprintf(u_sys,'%f\n', data);
    data = v(row,col);
    fprintf(v_sys,'%f\n', data);
end
close all;
end