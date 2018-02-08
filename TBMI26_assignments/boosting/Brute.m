function [kmin, T, P] = Brute(Xtrain, D, Ytrain)

[N, M] = size(Xtrain);


T = 1;
P = 1;
kmin = 1;

emin = inf;
for k = 1:N
    for i = 1:M
        tau = Xtrain(k, i);
        p = 1;
        C = WeakClassifier(tau, p, Xtrain(k, :));
        e = WeakClassifierError(C, D, Ytrain);
        if e > 0.5
            p = -1;
            e = 1 - e;
        end
        if e < emin
            emin = e;
            P = p;
            T = tau;
            kmin = k;
        end
    end
end

end
