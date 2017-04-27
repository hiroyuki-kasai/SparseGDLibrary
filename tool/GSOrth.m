% Modified Gram-Schmidt Orthonomalization
% (Source) http://elliottback.com/wp/modified-gram-schmidt-orthogonalization-in-matlab/
%
function Q = GSOrth(A)

n = size(A,2);
Q = A;

for k = 1:n-1,
    Q(:,k) = A(:,k) ./ norm(A(:,k));
    A(:,k+1:n) = A(:,k+1:n) - Q(:,k) * (Q(:,k)' * A(:,k+1:n));
end;

Q(:,n) = A(:,n) ./ norm(A(:,n));

end