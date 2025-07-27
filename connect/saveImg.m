function saveImg(ImageData)
    P = evalin('base','P');
    if P.saveImageFlag == 0
        return
    end

    fprintf("Saving...\n");

    bufferDir = 'C:\Users\verasonics\Desktop\Buffer';

    if ~exist(bufferDir, 'dir')
        mkdir(bufferDir);
    end

    % imaging
    ImageData(ImageData < 0) = 0; % negative clipping
    ImageData_log = log10(ImageData + 1);
    ImageData_norm = ImageData_log - min(ImageData_log(:));
    ImageData_norm = ImageData_norm / max(ImageData_norm(:));
    ImageData_uint8 = uint8(ImageData_norm * 255);

    % file naming
    dt = datetime('now');
    timestamp = char(string(dt, 'yyyyMMdd_HHmmss_SSS'));
    filename = sprintf('ultrasound_frame_%04d_%s.png', timestamp);
    filePath = fullfile(bufferDir, filename);

    imwrite(ImageData_uint8, filePath);
    fprintf("Save Complete!\n");
    return
end