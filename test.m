clc; clear; close all;

%% 1. 设置仿真区域与网格
L = 100;         % 空间边界长度
step = 1;        % 网格分辨率
[x, y] = meshgrid(0:step:L, 0:step:L);

%% 2. 初始化山峰障碍参数
n = 10;           % 山峰数量
Z = zeros(size(x));  % 初始化高度矩阵

% 随机生成每个山峰参数并叠加
rng(33);  % 固定随机种子保证可复现
for i = 1:n
    x_i = randi([20, 80]);     % 山峰中心 x
    y_i = randi([20, 80]);     % 山峰中心 y
    h_i = randi([15, 40]);     % 山峰高度
    x_si = randi([5, 15]);     % x方向坡度
    y_si = randi([5, 15]);     % y方向坡度
    
    % 累加高斯山峰
    Z = Z + h_i * exp(-((x - x_i)/x_si).^2 - ((y - y_i)/y_si).^2);
end

%% 3. 绘制三维地形
figure('Color','w');
surf(x, y, Z);              % 绘制地形表面
shading flat;               % 去除网格线
colormap(jet);              
colorbar;
xlabel('X (m)');
ylabel('Y (m)');
zlabel('Height (m)');
title('三维航迹规划环境模型（含10个山峰障碍）');
axis([0 100 0 100 0 60]);
view(45, 30);
hold on;

%% 4. 添加禁飞区（红色圆柱体表示）

% 禁飞区参数（可随机或自定义）
xc = 60; yc = 40;       % 禁飞区中心坐标
r_nf = 10;              % 禁飞区半径
h_nf = 50;              % 禁飞区高度（从地面起）

% 构造圆柱体边界
theta = linspace(0, 2*pi, 100);
X_cyl = xc + r_nf * cos(theta);
Y_cyl = yc + r_nf * sin(theta);
Z_bottom = zeros(size(theta));
Z_top = ones(size(theta)) * h_nf;

% 侧壁填充
for k = 1:length(theta)-1
    fill3([X_cyl(k) X_cyl(k+1) X_cyl(k+1) X_cyl(k)], ...
          [Y_cyl(k) Y_cyl(k+1) Y_cyl(k+1) Y_cyl(k)], ...
          [0 0 h_nf h_nf], ...
          'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none');
end

% 顶部盖板
fill3(X_cyl, Y_cyl, Z_top, 'r', 'FaceAlpha', 0.4, 'EdgeColor', 'none');

% 标注文字
text(xc, yc, h_nf + 2, 'No-Fly Zone', 'HorizontalAlignment', 'center', ...
    'Color', 'r', 'FontWeight', 'bold', 'FontSize', 10);

