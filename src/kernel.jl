# Randomized sketches
# https://arxiv.org/pdf/1501.06195.pdf

# Spectral truncation
# https://arxiv.org/pdf/1906.06276.pdf

# Backfitting
# https://www.stat.cmu.edu/~larry/=sml/nonpar.pdf


mutable struct KReg{T<:Real}

    # Responses
    y::Vector{T}

    # Covariates
    X::Matrix{T}

    # The kernel function
    ker::Kernel

    # The penalty parameter
    lam::Float64

    # The dimension of the sketch, or dimension to retain in the specral truncation.
    r::Int
end


mutable struct KGAM{T<:Real}

    # Responses
    y::Vector{T}

    # Each additive component of the GAM, represented as a kernel regression
    KR::Vector{KReg}
end


# Fit using spectral truncation.
function fit_spect(kr::KReg, krylov = true)

    (; y, X, ker, lam, r) = kr

    n = length(y)
    size(X, 1) == n || throw(error("length of y must equal the number of rows of x"))

    K = Symmetric(kernelmatrix(ker, X')) / n

    Ur, Dr = if krylov
        vals, vecs, info = eigsolve(z -> K * z, n, r, :LM)

        # eigsolve may return fewer than the reqeusted number of eigenvalues
        r = min(r, length(vals))
        Dr = real.(vals[1:r])
        Ur = zeros(n, r)
        for j = 1:r
            Ur[:, j] = real.(vecs[j])
        end
        Ur, Dr
    else
        a, b = eigen(K)
        ii = sortperm(a, rev = true)
        a = a[ii]
        b = b[:, ii]
        Dr = a[1:r]
        Ur = b[:, 1:r]
        Ur, Dr
    end

    m_alpha = (I(r) + lam * Diagonal(1 ./ Dr)) \ (Ur' / sqrt(n))
    alpha = m_alpha * y
    yhat = sqrt(n) * Ur * alpha

    kf = function (z)
        v = [ker(z, x) for x in eachrow(X)] / n
        v = sqrt(n) * Diagonal(1 ./ Dr) * Ur' * v
        return dot(alpha, v), v
    end

    return kf, yhat, m_alpha
end

# Fit using random sketching.
function fit_sketch(kr::KReg)

    (; y, X, ker, lam, r) = kr

    n = length(y)
    size(X, 1) == n || throw(error("length of y must equal the number of rows of x"))

    S = randn(n, r) / r^0.25

    KS = if n > 5000
        # This path is very slow but saves memory
        KS = zeros(n, r)
        for i = 1:n
            for j = 1:n
                kk = ker(X[i, :], X[j, :]) / n
                for k = 1:r
                    KS[i, k] += kk * S[j, k]
                end
            end
        end
        KS
    else
        K = Symmetric(kernelmatrix(ker, X')) / n
        KS = K * S
        KS
    end

    M = Symmetric(KS' * KS)
    F = Symmetric(S' * KS)

    m_alpha = ((M + 2 * lam * F) \ KS') / sqrt(n)
    alpha = m_alpha * y
    yhat = sqrt(n) * KS * alpha

    # Prediction function
    kf = function (z)
        v = [ker(z, x) for x in eachrow(X)] / n
        vs = sqrt(n) * S' * v
        return dot(vs, alpha), vs
    end

    return kf, yhat, m_alpha
end

function fit(kr::KReg; r::Int = -1, method::Symbol = :sketch, args...)
    if method == :sketch
        r = r < 0 ? 100 : r
        return fit_sketch(kr; args...)
    elseif method == :spect
        r = r < 0 ? 20 : r
        return fit_spect(kr; args...)
    else
        raise(error("unknown method"))
    end
end

function fit(kg::KGAM; maxiter = 10)

    (; y, KR) = kg

    ybar = mean(y)
    yc = y .- ybar

    n = length(y)
    fv = zeros(n, length(KR))

    for itr = 1:maxiter
        for j in eachindex(KR)
            jj = [k for k in eachindex(KR) if k != j]
            resid = yc - sum(fv[:, jj], dims = 2)
            KR[j].y .= resid
            _, fv[:, j], _ = fit(KR[j])
            fv[:, j] .-= mean(fv[:, j])
        end
    end

    yhat = ybar .+ sum(fv, dims = 2)
    return yhat, fv
end
