function res = posterior(lambda, t, tau)
    d = length(t) - 1;
    diff = zeros(d, 1);

    n = zeros(d, 1);
    
    for i = 1:d
        bla = find(tau >= t(i) & tau < t(i + 1));
        n(i) = length(bla);
    end

    for j = 1:d
        diff(j) = t(j + 1) - t(j);
    end
    
    res = exp(sum(log(lambda) .* n + log(diff) - lambda .* diff));
end