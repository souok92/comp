function saveImg2(ImageData)
    P = evalin('base','P');
    
    if (P.saveFrameFlag == 1 && P.saveImageFlag == 0) % save one by one
        switch P.gestureState
            case 1 % Fist
                fprintf("Save Fist\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Fist';
            case 2 % Index
                fprintf("Save Index\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Index';
            case 3 % Mid
                fprintf("Save Mid\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Mid';
            case 4 % Ring
                fprintf("Save Ring\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Ring';
            case 5 % Pinky
                fprintf("Save Pinky\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Pinky';
            case 6 % Open
                fprintf("Save Open\n");
                directory = 'C:\Users\verasonics\Desktop\Gest\Open';
        end
        saveImage(ImageData, directory);
        fprintf("Image Saved at %s", directory);

        P.saveFrameFlag = 0;
        assignin('base','P',P);
        return

    elseif (P.saveFrameFlag == 0 && P.saveImageFlag == 0)
        return

    else
        bufferDir = 'C:\Users\verasonics\Desktop\Buffer';
        saveImage(ImageData, bufferDir)
    return
    end
end

function saveImage(ImageData, Dir)
    if ~exist(Dir, 'dir')
        mkdir(Dir);
    end

    % imaging
    % ImageData(ImageData < 0) = 0; % negative clipping
    % ImageData_log = log10(ImageData + 1);
    % ImageData_norm = ImageData_log - min(ImageData_log(:));
    % ImageData_norm = ImageData_norm / max(ImageData_norm(:));
    % ImageData_uint8 = uint8(ImageData_norm * 255);
    
    % file naming
    dt = datetime('now');
    timestamp = char(string(dt, 'yyyyMMdd_HHmmss_SSS'));
    filename = sprintf('ultrasound_frame_%04d_%s.png', timestamp);
    filePath = fullfile(Dir, filename);

    imwrite(ImageData, filePath);
    fprintf("Save Complete!\n");
end