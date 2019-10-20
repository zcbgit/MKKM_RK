function index=indexTransRowwise(n,dims)
if n>prod(dims)
    error('index out of range');
end
dim=length(dims);
index=ones(dim,1);
for i=1:dim-1
    quotient=floor((n-1)/prod(dims(i+1:end))); % in case n=prod(dims)
    index(i)=quotient+1;
    remainder=rem(n,prod(dims(i+1:end)));
    if ~remainder
        index(i+1:end)=dims(i+1:end);
        return
    else
        n=rem(n,prod(dims(i+1:end)));
    end
end
index(dim)=n;