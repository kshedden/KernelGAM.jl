module KernelGAM

using LinearAlgebra, KrylovKit, KernelFunctions, Statistics

export KReg, KGAM, fit

include("kernel.jl")

end # module
