function [ x ] = det_Q( C,S )
    r=mod(C,S);     
    x=r>S./2;
end

