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

    # The fitted values
    yhat::Vector{T}

    # The coefficients of the linear predictor
    alpha::Vector{T}

    # Map from y to alpha
    y_alpha::Matrix{T}

    kreduce::Matrix{T}
end

function KReg(y, X, ker, lam, r)

    length(y) == size(X, 1) ||
        throw(error("Length of y must match the number of rows of X"))

    return KReg(y, X, ker, lam, r, zeros(0), zeros(0), zeros(0, 0), zeros(0, 0))
end


mutable struct KGAM{T<:Real}

    # Responses
    y::Vector{T}

    # Each additive component of the GAM, represented as a kernel regression
    KR::Vector{KReg}

    # Mean of the observed response values
    ybar::T

    # Fitted values
    yhat::Vector{T}

    # Fitted values for each additive component
    fitval::Matrix{T}
end

function KGAM(y::Vector, KR::Vector{KReg})
    n = length(y)
    q = length(KR)
    return KGAM(y, KR, 0.0, zeros(n), zeros(n, q))
end

# Fit using spectral truncation.
function fit_spect!(kr::KReg, krylov = true)

    (; y, X, ker, lam, r) = kr

    n = length(y)
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

    kr.y_alpha = (I(r) + lam * Diagonal(1 ./ Dr)) \ (Ur' / sqrt(n))
    kr.alpha = kr.y_alpha * y
    kr.yhat = sqrt(n) * Ur * kr.alpha
    kr.kreduce = sqrt(n) * Diagonal(1 ./ Dr) * Ur'
end

# predict with no arguments returns fitted values on the training data.
function predict(kr::KReg)
    return kr.yhat
end

function predict_wts(kr::KReg, x)
    n = size(kr.X, 1)
    v = [kr.ker(z, x) for z in eachrow(kr.X)] / n
    return kr.kreduce * v
end

function predict(kr::KReg, x)
    return kr.alpha' * predict_wts(kr, x)
end

# Fit using random sketching.
function fit_sketch!(kr::KReg)

    (; y, X, ker, lam, r) = kr

    n = length(y)
    S = randn(n, r) / r^0.25

    KS = if n > 10000
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

    kr.y_alpha = ((M + 2 * lam * F) \ KS') / sqrt(n)
    kr.alpha = kr.y_alpha * y
    kr.yhat = sqrt(n) * KS * kr.alpha
    kr.kreduce = sqrt(n) * S'
end

function fit!(kr::KReg; r::Int = -1, method::Symbol = :sketch, args...)
    if method == :sketch
        r = r < 0 ? 100 : r
        fit_sketch!(kr; args...)
    elseif method == :spect
        r = r < 0 ? 20 : r
        fit_spect!(kr; args...)
    else
        raise(error("unknown method"))
    end
end

function fit!(kg::KGAM; maxiter = 10)

    (; y, KR, fitval) = kg

    n = length(y)
    ybar = mean(y)
    kg.ybar = ybar
    yc = y .- ybar

    for itr = 1:maxiter
        for j in eachindex(KR)
            jj = [k for k in eachindex(KR) if k != j]
            resid = yc - sum(fitval[:, jj], dims = 2)
            KR[j].y .= resid
            fit!(KR[j])
            fitval[:, j] = predict(KR[j])
            fitval[:, j] .-= mean(fitval[:, j])
        end
    end

    kg.yhat .= ybar .+ sum(fitval, dims = 2)[:]
end

function predict(kg::KGAM)
    return kg.yhat
end
