module KernelGAM

using LinearAlgebra, KrylovKit, KernelFunctions, Statistics

export KReg, KGAM, fit!, predict, predict_wts

include("kernel.jl")

end
