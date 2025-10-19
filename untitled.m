[x, y] = meshgrid(0:1:100, 0:1:100);
z = 50 * sin(0.1 * x) .* cos(0.1 * y) + 100 * rand(size(x));
figure;
surf(x, y, z);
colormap('jet');
shading interp;
