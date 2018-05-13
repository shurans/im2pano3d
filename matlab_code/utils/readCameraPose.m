function cameraPoses = readCameraPose(camereafile)
cameraPoses =[];
fid = fopen(camereafile,'r');
tline = fgets(fid);
while ischar(tline)
    parseline = sscanf(tline, '%f');
    if size(parseline,1)~=12
        break;
    end
    cameraPoses = [cameraPoses;parseline'];
    
    tline = fgets(fid);
end
fclose(fid);
end