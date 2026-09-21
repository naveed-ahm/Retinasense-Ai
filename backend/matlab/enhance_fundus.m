function resultJson = enhance_fundus(imagePath, outputPath)
%ENHANCE_FUNDUS Create a visualization-only enhanced copy of a fundus image.
% This routine never changes imagePath and is not used by disease inference.

    if strcmpi(char(imagePath), char(outputPath))
        error('enhance_fundus:InvalidOutput', 'Output path must not overwrite the original input image.');
    end

    image = im2double(imread(char(imagePath)));
    if size(image, 3) == 3
        green = image(:, :, 2);
    else
        green = image;
    end

    % Estimate slow illumination variation, then normalize before local CLAHE.
    illumination = imgaussfilt(green, max(15, round(min(size(green)) / 20)));
    normalized = mat2gray(green ./ max(illumination, 0.05));
    contrasted = adapthisteq(normalized, 'NumTiles', [8 8], 'ClipLimit', 0.01);
    denoised = imgaussfilt(contrasted, 0.5);
    enhancedGreen = imsharpen(denoised, 'Radius', 1, 'Amount', 0.5);

    if size(image, 3) == 3
        % Retain colour context while using the enhanced green channel for detail.
        hsvImage = rgb2hsv(image);
        hsvImage(:, :, 3) = mat2gray(0.65 * hsvImage(:, :, 3) + 0.35 * enhancedGreen);
        enhanced = hsv2rgb(hsvImage);
    else
        enhanced = enhancedGreen;
    end

    out = char(outputPath);
    outDir = fileparts(out);
    if ~isempty(outDir) && ~isfolder(outDir)
        mkdir(outDir);
    end
    imwrite(enhanced, out);
    resultJson = jsonencode(struct('enhancement_path', out));
end
