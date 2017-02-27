function out = resizeFlow(flow, sz)
    [M, N, C] = size(flow);
    
    facty = M/sz(1);
    factx = N/sz(2);
    
    u = imresize(flow(:,:,1), sz)/factx;
    v = imresize(flow(:,:,2), sz)/facty;
        
    out(:,:,1) = u;
    out(:,:,2) = v;
