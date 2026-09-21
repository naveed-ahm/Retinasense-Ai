function resultJson = retinal_features(imagePath)
%RETINAL_FEATURES Non-diagnostic, image-derived summary measurements only.

    image = im2double(imread(char(imagePath)));
    if size(image, 3) == 3
        green = image(:, :, 2);
    else
        green = image;
    end
    fovMask = imfill(imclose(green > 0.04, strel('disk', 8)), 'holes');
    if nnz(fovMask) < 100
        error('retinal_features:InvalidImage', 'Image does not contain a usable field.');
    end
    pixels = green(fovMask);
    contrastValue = std(pixels);
    background = imgaussfilt(green, 12);
    vesselLike = (background - green) > 0.06 & fovMask;
    vesselLike = bwareaopen(vesselLike, 8);
    result = struct('mean_intensity', round(mean(pixels), 4), ...
                    'contrast', round(contrastValue, 4), ...
                    'bright_region_percentage', round(100 * mean(pixels > 0.80), 2), ...
                    'dark_region_percentage', round(100 * mean(pixels < 0.15), 2), ...
                    'retinal_field_area', round(100 * nnz(fovMask) / numel(green), 2), ...
                    'vessel_density', round(100 * nnz(vesselLike) / nnz(fovMask), 2));
    resultJson = jsonencode(result);
end
