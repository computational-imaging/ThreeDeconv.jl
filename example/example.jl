using PyCall, Printf
@pyimport skimage.io as io
using ThreeDeconv
using CuArrays

function imread(filename::String)
    img = io.imread(filename)
    if ndims(img)==3
        img = permutedims(img, (2,3,1))
    end
    return img
end

function imsave(filename::String, img::Array)
    if ndims(img)==3
        img = permutedims(img, (3,1,2))
    end
    io.imsave(filename, img)
end

imgraw = imread("dataset/raw/bead/raw_cropped/bead_highsnr_raw.tif")
df = imread("dataset/raw/bead/raw_cropped/df.tif")
ff = imread("dataset/raw/bead/raw_cropped/ff_highsnr.tif")
img = (imgraw .- df) ./ ff

obj_mag = 100
camera_pixel_size = 6.5e3 # [nm]
xystep = camera_pixel_size / obj_mag
zstep = 150 # [nm]
medium_index = 1.515
f_tubelens = 200.e6
NA = 1.4
λ = 540 # [nm]


γ, σ = ThreeDeconv.noise_estimation(img, maxnum_pairs=100)
@printf "Gain: %.1f, Read noise std.: %.1f" γ σ

pad = 10
psf_shape = size(img) .+ pad
println("Simulating PSF with the size of $(psf_shape)")
psf = ThreeDeconv.psf(
    NA = NA,
    objective_mag = obj_mag,
    λ = λ,
    medium_index = medium_index,
    psf_shape = psf_shape,
    camera_pixel_size = camera_pixel_size,
    zstep = zstep,
    f_tubelens = f_tubelens,
    oversampling = 4)


options = ThreeDeconv.DeconvolutionOptions(
    max_iters=150,
    show_trace=true,
    check_every=10
)
reg = 0.01
result = ThreeDeconv.deconvolve(img, psf, γ, σ, reg, options=options);

imsave("deconvoled.tif", result.x)
