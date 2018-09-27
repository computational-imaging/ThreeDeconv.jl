# ThreeDeconv v1.0 release note
This package is mostly developed by [@hayatoikoma](https://github.com/hayatoikoma), and the PSF simulator (WaveOptics.jl) is developed by [@broxtronix](https://github.com/broxtronix).
`WaveOptics.jl` is still an unreleased Julia package for various PSF simulations.
`ThreeDeconv.jl` has incorporated and modified part of `WaveOptics.jl` for this release.

Some of the major changes we made are
* Use of `CuArrays.jl`, instead of `ArrayFire.jl`, for GPU computing.
* Reduce the number of memory transfter between CPU and GPU. (Now, the computation time for Fig.2 is 26.2 sec.)
* Speed up of a noise estimation.
    * In the paper, we were using all local mean-variance pairs for the maximum likelihood estimation in Foi's algorithm. But, this package uses a subset of the local pairs for the final noise parameter estimation, which has significantly reduced the estimation time.
* Speed up of a PSF simulation. (Now we are computing the PSF at a focal plane)
