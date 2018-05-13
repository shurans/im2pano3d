function image = autoCropImage(image)
% input an image
% output an image with white space removed
% assuming the top left pixel is the background color that you don't want
if ischar(image)
    imagename = image;
    image = im2double(imread(image));
else
    imagename = [];
end

mask = mean(image,3);
mask = mask == mask(1,1);
isGood1 = find(any(~mask,1));
isGood2 = find(any(~mask,2));
image = image(min(isGood2):max(isGood2),min(isGood1):max(isGood1), :);

if ~isempty(imagename)
   imwrite(image,imagename) 
end