clear
close all

addpath(genpath('../matlab_toolboxes'))

% Load stuff
wsurf = load('../data/wsurf.mat');
walong = load('../data/walong.mat');
wacross = load('../data/wacross.mat');

f = figure(11);
bcolor = [0.8902    0.8824    0.8706]; % background color, used below
width = 5;  % inches
height = 4;  % inches
f.Units = 'Inches';
f.Position = [3, 3, width, height];
f.PaperPositionMode = 'Manual';
f.PaperSize = [width, height];

% caxis
clims = [-0.05, 0.05];

% surface
[xg, yg] = meshgrid(wsurf.x, wsurf.y);
zg = -wsurf.z*ones(size(xg));
wplot = wsurf.w;
wplot(wsurf.w == 0) = nan;
s = surf(xg, yg, zg, wplot);
s.EdgeColor = 'none';
caxis(clims)
colormap(cmocean('balance'))

cb = colorbar('location', 'north');
cb.Position = [0.1 0.8 0.3 0.02];
xlabel(cb, 'w [m s^-^1]')

hold on

% along slice
[xg, zg] = meshgrid(walong.x, -walong.z);
yg = walong.y*ones(size(xg));
wplot = walong.w;
wplot(walong.w == 0) = nan;
s = surf(xg, yg, zg, wplot);
s.EdgeColor = 'none';
caxis(clims)

% glacier
fc = 0.8;
patch([17.5 17.5 17.5 17.5], [400 990 990 400],  [-145 -145 0 0], [fc fc fc])
patch([17.5 0 0 17.5], [400 400 990 990],  [-145 -161 -161 -145], [fc fc fc])

% across slice
[yg, zg] = meshgrid(wacross.y, -wacross.z);
xg = wacross.x*ones(size(yg));
s = surf(xg, yg, zg, wacross.w);
s.EdgeColor = 'none';
caxis(clims)

% locations of mooring points
scatter3(142, 600, -48,'MarkerEdgeColor', 'k',...
              'MarkerFaceColor','y',...
              'LineWidth', 2)

text(142, 600, -10, 'MN*', ...
    'BackgroundColor', bcolor, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle')

scatter3(500, 600, -48,'MarkerEdgeColor', 'k',...
              'MarkerFaceColor','y',...
              'LineWidth', 2)

text(500, 600, -10, 'MD*', ...
    'BackgroundColor', bcolor, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle')

yticks([200 400 600 800 1000])

angle = 160;
pbaspect([3, 3, 1]);
view(angle, 20);

set(gca, 'YDir','reverse', 'Box','off')
zlim([-165, 0])
xlim([0, 600])
ylim([200, 1000])

xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')

exportgraphics(f, '../figures/3D_model_matlab.pdf', 'Resolution', 300)
exportgraphics(f, '../figures/3D_model_matlab.png', 'Resolution', 300)