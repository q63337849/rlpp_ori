function path=plotFigure_rect(data,LegendStr,strcolor)
global N startPos goalPos boxes mapRange
num = numel(data);

% 生成每条路径
for i=1:num
    x     = data(i).Best_pos;
    x_seq = [startPos(1), x(1:N),           goalPos(1)];
    y_seq = [startPos(2), x(N+1:2*N),       goalPos(2)];
    z_seq = [startPos(3), x(2*N+1:3*N),     goalPos(3)];
    k     = length(x_seq);
    I_seq = linspace(0,1,100);
    X_seq = spline(linspace(0,1,k), x_seq, I_seq);
    Y_seq = spline(linspace(0,1,k), y_seq, I_seq);
    Z_seq = spline(linspace(0,1,k), z_seq, I_seq);
    path(i).data = [X_seq', Y_seq', Z_seq'];
end

figure; hold on; box on; grid on
axis([0 mapRange(1) 0 mapRange(2) 0 mapRange(3)])
xlabel('x'); ylabel('y'); zlabel('z'); view(3)

% 画长方体障碍
for j=1:size(boxes,1)
    draw_box(boxes(j,:), 0.3); % 透明度
end

% 画起点/终点
scatter3(startPos(1), startPos(2), startPos(3), 60, 'g', 'filled')
text(startPos(1), startPos(2), startPos(3)+6, '起点')
scatter3(goalPos(1), goalPos(2), goalPos(3), 60, 'r', 'filled')
text(goalPos(1), goalPos(2), goalPos(3)+6, '终点')

% 画路径
leg = gobjects(1,num);
for i=1:num
    leg(i) = plot3(path(i).data(:,1), path(i).data(:,2), path(i).data(:,3), ...
                   strcolor{i}, 'LineWidth', 2.0);
end
legend(leg, LegendStr, 'location','best')
set(gcf,'color','w')
end

function draw_box(b, alphaV, fc)
% b = [x y z w l h]  轴对齐长方体
if nargin < 3, fc = [0.8 0.2 0.2]; end   % 颜色可选
x0 = b(1); y0 = b(2); z0 = b(3);
x1 = x0 + b(4); 
y1 = y0 + b(5); 
z1 = z0 + b(6);

% 8 个顶点（顺序固定）
V = [ ...
    x0 y0 z0;  % 1
    x1 y0 z0;  % 2
    x1 y1 z0;  % 3
    x0 y1 z0;  % 4
    x0 y0 z1;  % 5
    x1 y0 z1;  % 6
    x1 y1 z1;  % 7
    x0 y1 z1]; % 8

% 6 个面（每行是一个四边形面的顶点索引）
F = [ ...
    1 2 3 4;   % 底面 z=z0
    5 6 7 8;   % 顶面 z=z1
    1 2 6 5;   % 前面 y=y0
    2 3 7 6;   % 右面 x=x1
    3 4 8 7;   % 后面 y=y1
    4 1 5 8];  % 左面 x=x0

patch('Vertices',V,'Faces',F, ...
      'FaceColor',fc, 'EdgeColor',[0.3 0.3 0.3], ...
      'FaceAlpha',alphaV);
end

