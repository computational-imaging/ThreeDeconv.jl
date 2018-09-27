include("WaveOptics.jl")

function psf(; objective_mag, NA, λ, medium_index, f_tubelens, camera_pixel_size, camera_num_pixels, z_num_pixels, zstep, oversampling)
    NA = NA > medium_index ? medium_index : NA
    recipe = WaveOptics.WideFieldOpticalRecipe(Float64(objective_mag), Float64(NA), Float64(λ), Float64(medium_index), Float64(f_tubelens))

    sim_size_px = camera_num_pixels * oversampling
    sim_size_m = camera_num_pixels * camera_pixel_size / objective_mag

    model = WaveOptics.DefocusModel(recipe, sim_size_m, sim_size_px)

    # Compute the 3D PSF at the sampling resolution
    zmax = zstep * (z_num_pixels - 1) * 0.5
    zrange = range(-zmax, step=zstep, length=z_num_pixels)

    psf = WaveOptics.psf_3d(model, zrange)
    psf_sensor = WaveOptics.downsample_to_sensor_resolution(psf, oversampling)
    factor = sum(psf_sensor[:,:,div(z_num_pixels,2)])

    return psf_sensor ./ factor
end
