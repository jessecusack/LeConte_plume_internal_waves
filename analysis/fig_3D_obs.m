% This code makes a 3D plot of the LeConte near-glacier region
% The data plotted first has to be created by running
% fig_observational_overview.ipynb
% The matlab toolboxes must be downloaded by running the shell script in
% the matlab_toolboxes directory.
clear
close all

addpath(genpath('../matlab_toolboxes'))

bathy = load('../data/bathy.mat');
% Loads the terminus point clouds
load('../data/combo_icexyz_Sept2018.mat');
MD = load('../data/MD.mat');
MN = load('../data//MN.mat');

% pre plotting stuff
x0 = nanmin(bathy.x,[],'all') + 600;  
% The 600 is added to get a similar origin to the model plot
y0 = nanmin(bathy.y,[],'all') - 400;

f = figure(11);
% bcolor = [0.8902    0.8824    0.8706]; % background color, used below
bcolor = [0.95    0.95    0.95];
width = 5;  % inches
height = 4;  % inches
f.Units = 'Inches';
f.Position = [3, 3, width, height];
f.PaperPositionMode = 'Manual';
f.PaperSize = [width, height];

% bathymetry
tiledlayout(1, 1,'TileSpacing','Compact','Padding', 'none');
nexttile
s = surf(-(bathy.x - x0), bathy.y - y0, -bathy.H);
s.EdgeColor = 'none';
s.AmbientStrength = 0.8;
% cmap = cmocean('topo', 200);
% colormap(cmap(1:100, :));
colormap(cmocean('gray')); % gray
caxis([-300, 0]);
shading interp
camlight(90, 50)
set(gca, 'AmbientLightColor', [0.5, 1, 1])
% hbar = colorbar('horiz');
% xlabel(hbar, 'Depth [m]')
freezeColors

hold on

% bathymetry contours
contour3(-(bathy.x - x0), bathy.y - y0, -bathy.H, -250:25:-25, 'color', 'k', 'linewidth', 1)

% terminus
step = 3;
ice = ice(1);
% msize = 20;
% scatter3(ice.x(1:step:end) - x0, ice.y(1:step:end) - y0, -ice.z(1:step:end), msize, -ice.z(1:step:end), 'filled');
% colormap(cmocean('ice'));
% caxis([-200, 0]);

p = [-(ice.x(1:step:end) - x0) ice.y(1:step:end) - y0 -ice.z(1:step:end)];
% keep = ice.y(1:step:end) > y0 + 50;
% p = p(keep, :);

t = MyCrustOpen(p);
ts = trisurf(t, p(:,1), p(:,2), p(:,3), 10*ones(size(p(:, 1))), 'edgecolor', 'none');
ts.AmbientStrength = 0.7;
shading interp
%colormap('gray'); % cmocean('ice')
caxis([-200, 0]);
% set(ts,'FaceAlpha', 0.5);

% mooring plot properties
lw = 5;
color = [0.95    1.0    0.05];
ls = '-';
step = 3;
qs = 1000;  % manual scale factor
qcolor = [0.9290 0.6940 0.1250];%0.8*color; %[0    0.4471    0.7412];
qlw = 3;
msize = 10;
mfcolor = 0.8*color;
SAH = 'off';

% MD mooring plot
plot3(-(MD.x - x0), MD.y - y0, MD.z, 'color', color, 'linestyle', ls, 'linewidth', lw)
quiver3(-(MD.x(1:step:end) - x0), MD.y(1:step:end) - y0, MD.z(1:step:end), ...
        -qs*MD.u(1:step:end), qs*MD.v(1:step:end), qs*MD.w(1:step:end), ...
        'AutoScale', false, 'color', qcolor, 'linewidth', qlw, 'ShowArrowHead', SAH)
       
text(double(-(MD.x(1) - x0)), double(MD.y(1) - y0), MD.z(1) + 50, 'MD', ...
    'BackgroundColor', bcolor, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle')
% set(t, 'FaceAlpha', 0.5)

% MN mooring plot
plot3(-(MN.x - x0), MN.y - y0, MN.z, 'color', color, 'linestyle', ls, 'linewidth', lw)
quiver3(-(MN.x(1:step:end) - x0), MN.y(1:step:end) - y0, MN.z(1:step:end), ...
        -qs*MN.u(1:step:end), qs*MN.v(1:step:end), qs*MN.w(1:step:end), ...
        'AutoScale', false, 'color', qcolor, 'linewidth', qlw, 'ShowArrowHead', SAH)

text(double(-(MN.x(end) - x0)), double(MN.y(end) - y0), MN.z(end) + 50, 'MN', ...
    'BackgroundColor', bcolor, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle')

% plot viewing stuff and labels
angle = 160;
pbaspect([3, 3, 1]);
view(angle, 35);
set(gca, 'YDir','reverse')


% Quiver key hack
set(gca,'Color', bcolor)
spd = 0.05;
xlims = get(gca, 'xlim');
ylims = get(gca, 'ylim');

xqk = 0.8*xlims(2);
yqk = ylims(2);
zqk = -70;

quiver3(xqk, yqk, zqk, qs*spd*cosd(angle), ...
    -qs*spd*sind(angle), 0, 'AutoScale', false, 'color', qcolor, ...
    'linewidth', qlw, 'ShowArrowHead', SAH)

text(xqk + 50, yqk, zqk + 50, sprintf('%1.2f m s^{-1}', spd))

xlabel('x [m]')
ylabel('y [m]')
zlabel('z [m]')

exportgraphics(f, '../figures/3D_obs_matlab.pdf', 'Resolution', 300)
exportgraphics(f, '../figures/3D_obs_matlab.png', 'Resolution', 300)