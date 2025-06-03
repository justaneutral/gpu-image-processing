D = dir('*.bmp')
if length(D)>0
    s = imread(D(1).name);
    imwrite(s,'Lena.pgm','pgm');
end
