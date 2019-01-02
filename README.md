# ThreeDeconv.jl

**ThreeDeconv.jl** is a 3D deconvolution software for fluorescence microscopy written in [Julia](https://julialang.org).
The detail of the algorithm is available in [our paper](https://www.nature.com/articles/s41598-018-29768-x#Sec21) and [our website](http://www.computationalimaging.org/publications/2d-deconvolution-for-low-photon-count-fluorescence-imaging-scientific-reports-2019/).
While the deconvolution algorithm is the same as described in the paper, we made some improvements in our software along with the major update of Julia and related packages. Thanks to this update, this package is significantly faster than the speed reported on the paper and deconvoles a 256 x 256 x 57 image in 5 seconds!  Please refer to [NEWS.md](NEWS.md) to catch up the updates.

# Installation

Hit the key `]` in the Julia REPL to ender the Pkg REPL-mode, and run

```julia-repl
pkg> add https://github.com/computational-imaging/ThreeDeconv.jl.git
```

# Example usage
To run the example script, you need [`PyCall.jl`](https://github.com/JuliaPy/PyCall.jl) and [`Conda.jl`](https://github.com/JuliaPy/Conda.jl). In addition, you need [scikit-image](https://scikit-image.org/) available through `PyCall`. You can run the following in Julia's REPL to install them.
```julia-repl
pkg> add Conda
julia> using Conda
julia> Conda.add_channel("conda-forge")
julia> Conda.add("scikit-image")
julia> ENV["PYTHON"]=""
pkg> add PyCall 
```
If you already have a Python environment that you want to use from Julia, please consult with `PyCall.jl`.
Since these packages are required just for image I/O, you can also use other packages if you want.
However, I personally found that `scikit-image` is the most convenient package to read multi-stack tiff files from Julia.

If you have a CUDA-compatible NVIDIA GPU, you can use [`CuArrays.jl`](https://github.com/JuliaGPU/CuArrays.jl) to accelerate deconvolution by just running `using CuArrays` after loading `ThreeDeconv.jl`. You can install `CuArrays` by running the following command in Julia's REPL.
```julia-repl
pkg> add CuArrays
```

Then, after downloading the dataset (described below) in the `example` directory, run
```
julia example.jl
```
from your terminal in the `example` directory. The script simulates the PSF, deconvolves the fluorescent bead and saves the deconvolved image.

The sample script is actually a script converted from [the Jupyter notebook](https://github.com/computational-imaging/ThreeDeconv.jl/blob/master/example/Example.ipynb) also available in the same directory. To run the notebook, you need [`IJulia.jl`](https://github.com/JuliaLang/IJulia.jl) and [`PyPlot.jl`](https://github.com/JuliaPy/PyPlot.jl). 


# Dataset
The dataset used in our paper is available at [Google Drive (150MB)](https://drive.google.com/a/stanford.edu/file/d/1lWlvngb5iJkToFKLSA3N1FuScVPTe-42/view?usp=sharing).
The images are processed with darkfield and flatfield correction as described in the paper. Its raw dataset is also available at [Google Drive (1GB)](https://drive.google.com/a/stanford.edu/file/d/1pg_OG5GxjcKMSvwi4Si0HTyWT0XCW4Kt/view?usp=sharing).
This dataset comes with corresponding darkfield and flatfield images.

This dataset can be used for academic and other non-commercial purposes.
If you use this dataset in your research paper, please cite our paper.



# Citation
["A convex 3D deconvolution algorithm for low photon count fluorescence imaging"](https://www.nature.com/articles/s41598-018-29768-x#Sec21)
_Scientific Reports_ **8**, Article number: 11489 (2018)
Hayato Ikoma, Michael Broxton, Takamasa Kudo, Gordon Wetzstein

```
@article{ikoma2018convex,
  title={A convex 3D deconvolution algorithm for low photon count fluorescence imaging},
  author={Ikoma, Hayato and Broxton, Michael and Kudo, Takamasa and Wetzstein, Gordon},
  journal={Scientific reports},
  volume={8},
  number={1},
  pages={11489},
  year={2018},
  publisher={Nature Publishing Group}
}
```

If you use this library for your research, please cite our paper.


# To biologists

If you find this package useful but feel that it lacks some functionalities for your research, feel free to create an issue in this repository or send an email [@hayatoikoma](https://github.com/hayatoikoma). I am happy to extend this package to accomodate your need.

# Developers
This package is mostly developed by [@hayatoikoma](https://github.com/hayatoikoma), and the PSF simulator (WaveOptics.jl) is developed by [@broxtronix](https://github.com/broxtronix).
`WaveOptics.jl` is still an unreleased Julia package for various PSF simulations.
`ThreeDeconv.jl` has incorporated part of `WaveOptics.jl` for this release.


# License
The ThreeDeconv.jl package is licensed under the following license:

> Copyright (c) 2018, Stanford University
>
> All rights reserved.
>
> Redistribution and use in source and binary forms for academic and other non-commercial purposes with or without modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code, including modified source code, must retain the above copyright notice, this list of conditions and the following disclaimer.
>
> * Redistributions in binary form or a modified form of the source code must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
>
> * Neither the name of The Leland Stanford Junior University, any of its trademarks, the names of its employees, nor contributors to the source code may be used to endorse or promote products derived from this software without specific prior written permission.
>
> * Where a modified version of the source code is redistributed publicly in source or binary forms, the modified source code must be published in a freely accessible manner, or otherwise redistributed at no charge to anyone requesting a copy of the modified source code, subject to the same terms as this agreement.
>
> THIS SOFTWARE IS PROVIDED BY THE TRUSTEES OF THE LELAND STANFORD JUNIOR UNIVERSITY "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE LELAND STANFORD JUNIOR UNIVERSITY OR ITS TRUSTEES BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
