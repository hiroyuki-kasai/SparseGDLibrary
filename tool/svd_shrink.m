function [ v ] = svd_shrink(w, t, dim)
    if nargin == 3
        L = reshape(w, dim);
    else
        L = w;
    end
    [U,S,V] = svd(L, 'econ');
    s = diag(S);
    S = diag(sign(s) .* max(abs(s) - t,0));
    L = U*S*V';
    if nargin == 3
        v = L(:);
    else
        v = L;
    end
end

