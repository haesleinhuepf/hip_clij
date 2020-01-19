clear;

% init and test HIP
import HIP.*
HIP.Cuda.DeviceStats()

% init and test CLIJ
clijx = init_clatlab();
clijx.getGPUName()
clijx.clear();

for image_size = 100:50:350
    % -------------------------------------------------------------------------
    % define example data
    %filename = 'blobs.tif';
    %input_image = imread(filename);
    %input_image = double(input_image);

    %image_size = 1000;
    input_image = zeros(image_size, image_size, image_size);
    input_image(image_size / 2, image_size / 2, image_size / 2) = 1000000;

    number_of_iterations = 1;
    sigma = 50;

    % -------------------------------------------------------------------------
    % test HIP
    hip_time_start = now;
    hip_smoothed_image = HIP.Gaussian(input_image, [sigma, sigma, sigma], number_of_iterations, [ ] );
    % measure duration
    hip_duration = now - hip_time_start;

    % show result
    subplot(1,3,1), imshow(hip_smoothed_image(:,:,image_size/2));

    % -------------------------------------------------------------------------
    % test CLIJ
    clij_time_start = now;
    clij_input_image = clijx.pushMat(input_image);
    clij_smoothed_image = clijx.create(clij_input_image);
    clijx.blur3D(clij_input_image, clij_smoothed_image, sigma, sigma, sigma);
    clij_result = clijx.pullMat(clij_smoothed_image);
    clij_duration = now - clij_time_start;

    % for debugging:
    %clijx.show(clij_smoothed_image, "clij_smoothed_image");

    % show result
    subplot(1,3,2), imshow(clij_result(:,:,image_size/2));

    % -------------------------------------------------------------------------
    % measure differences
    clij_hip_result = clijx.pushMat(hip_smoothed_image);
    mse_hip_clij = clijx.meanSquaredError(clij_hip_result, clij_smoothed_image);


    clijx.clear();

    % show differences
    difference = hip_smoothed_image - clij_result;
    subplot(1,3,3), imshow(difference(:,:,image_size/2));

    % -------------------------------------------------------------------------
    % Output measurements
    fprintf('Image size: %dx%dx%d\n', image_size, image_size, image_size);
    fprintf('HIP duration (MM:SS.FFF): %s\n',  datestr(hip_duration,'MM:SS.FFF'));
    fprintf('CLIJ duration (MM:SS.FFF): %s\n',  datestr(clij_duration,'MM:SS.FFF'));
    fprintf('MSE: %f\n-------------------\n', mse_hip_clij);
end


