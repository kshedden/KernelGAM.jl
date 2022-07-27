using KernelGAM, Test, KernelFunctions, StableRNGs, Suppressor

@testset "basic" begin

    n = 1000
    p = 5
    nrep = 100
    r = Dict(:sketch => 100, :spect => 30)
    rng = StableRNG(123)

    for method in [:spect]
        for kern in [:sqexp, :poly, :sqexp]
            ker, lam = if kern == :sqexp
                with_lengthscale(SqExponentialKernel(), 3.0), 1e-5
            elseif kern == :poly
                ker = with_lengthscale(PolynomialKernel(degree = 2, c = 1.0), 0.5), 1e-4
            else
                error("unknown kernel")
            end
            mse = 0.0
            for i = 1:nrep
                X = 2 * rand(rng, n, p) .- 1
                X[:, 2] = sort(X[:, 2])
                ey = 2 * X[:, 2] .^ 2
                y = ey + randn(rng, n)

                kr = KReg(y, X, ker, lam, r[method])
                yhat = nothing
                @suppress begin
                    _, yhat = fit(kr; method = method)
                end

                mse += sum(abs2, yhat - ey) / n
            end
            mse /= nrep
            rmse = sqrt(mse)
            @test rmse < 0.2
        end
    end
end

@testset "prediction" begin

    n = 1000
    p = 5
    r = 20
    bw = 2.0
    lam = 1e-4
    rng = StableRNG(123)

    for method in [:spect, :sketch]
        for kern in [:sqexp, :poly]
            ker = if kern == :sqexp
                with_lengthscale(SqExponentialKernel(), bw)
            elseif kern == :poly
                ker = with_lengthscale(PolynomialKernel(degree = 2, c = 1.0), bw)
            else
                error("unknown kernel")
            end
            X = 2 * rand(rng, n, p) .- 1
            X[:, 2] = sort(X[:, 2])
            ey = 2 * X[:, 2] .^ 2
            y = ey + randn(rng, n)

            kr = KReg(y, X, ker, lam, r)
            f, yhat, _ = fit(kr; method = method)

            yhat2 = [f(x)[1] for x in eachrow(X)]
            @test isapprox(yhat, yhat2)
        end
    end
end

@testset "backfit" begin
    n = 1000
    rng = StableRNG(123)
    X = randn(rng, n, 3)
    ey = X[:, 1] - X[:, 2] .* X[:, 3]
    y = ey + randn(rng, n)

    r = 20
    bw = 3.0
    ker = with_lengthscale(SqExponentialKernel(), bw)
    ker = with_lengthscale(PolynomialKernel(degree = 4, c = 1.0), bw)
    lam = 1e-4

    KR = KReg[]
    push!(KR, KReg(y, X[:, 1:1], ker, lam, r))
    push!(KR, KReg(y, X[:, 2:3], ker, lam, r))

    kg = KGAM(y, KR)
    yhat, fv = fit(kg)

    rmse1 = sqrt(sum(abs2, X[:, 1] - fv[:, 1]) / n)
    rmse2 = sqrt(sum(abs2, -X[:, 2] .* X[:, 3] - fv[:, 2]) / n)

    @test rmse1 < 0.1
    @test rmse2 < 0.1
end
