function resultJson = fundus_quality(imagePath)
%FUNDUS_QUALITY Non-diagnostic engineering quality checks for a fundus image.
% Scores are 0-100 and use image-quality, not clinical, thresholds:
% brightness: best mean gray intensity is 0.35-0.75; contrast: std >= 0.12;
% sharpness: variance of the Laplacian reaches full score at 0.002; and
% FOV: a connected, non-dark retinal field covering >= 35% of the frame.

    image = im2double(imread(char(imagePath)));
    if size(image, 3) == 3
        gray = rgb2gray(image);
    else
        gray = image;
    end

    valid = gray > 0.04;
    if nnz(valid) < 100
        error('fundus_quality:InvalidImage', 'Image does not contain a usable field.');
    end
    pixels = gray(valid);
    meanIntensity = mean(pixels);
    contrastValue = std(pixels);

    % Laplacian variance is a standard, image-derived blur/sharpness metric.
    lap = imfilter(gray, [0 1 0; 1 -4 1; 0 1 0], 'replicate');
    sharpnessValue = var(lap(valid));

    % Retinal FOV is approximated by the largest non-dark connected region.
    mask = imclose(valid, strel('disk', 8));
    mask = imfill(mask, 'holes');
    components = bwconncomp(mask);
    areas = cellfun(@numel, components.PixelIdxList);
    fovFraction = 0;
    if ~isempty(areas)
        fovFraction = max(areas) / numel(gray);
    end

    brightnessScore = clamp01(1 - abs(meanIntensity - 0.55) / 0.35) * 100;
    contrastScore = clamp01(contrastValue / 0.12) * 100;
    sharpnessScore = clamp01(sharpnessValue / 0.002) * 100;
    fovScore = clamp01((fovFraction - 0.15) / 0.35) * 100;
    qualityScore = round(0.25 * brightnessScore + 0.25 * contrastScore + ...
                         0.30 * sharpnessScore + 0.20 * fovScore);

    if qualityScore >= 75
        status = 'GOOD';
    elseif qualityScore >= 50
        status = 'ACCEPTABLE';
    else
        status = 'POOR';
    end

    result = struct('quality_score', round(qualityScore), ...
                    'brightness_score', round(brightnessScore), ...
                    'contrast_score', round(contrastScore), ...
                    'sharpness_score', round(sharpnessScore), ...
                    'fov_score', round(fovScore), ...
                    'status', status);
    resultJson = jsonencode(result);
end

function value = clamp01(value)
    value = min(max(value, 0), 1);
end
