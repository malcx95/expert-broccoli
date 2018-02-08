function [H, e] = StrongClassifier(X, Alpha, T, P, K)

Ctot = 0;
for i = 1:length(T)
    C = WeakClassifier(T(i), P(i), X(K(i), :));
    Ctot = Ctot + Alpha(i)*C;
end
%H = sign(sum(Alpha.*C));

H = sign(Ctot);
if nargout == 2
    e = 0;
end

end
