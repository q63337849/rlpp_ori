function fitness = CostRect(x)
% 路径长度 + 碰撞罚项 + 越界罚项
global N startPos goalPos mapRange boxes

% 1) 三次样条生成离散路径点（与原来一致）
x_seq = [startPos(1), x(1:N),           goalPos(1)];
y_seq = [startPos(2), x(N+1:2*N),       goalPos(2)];
z_seq = [startPos(3), x(2*N+1:3*N),     goalPos(3)];
k     = length(x_seq);
i_seq = linspace(0,1,k);
I_seq = linspace(0,1,100);              % 采样更密一点更稳妥
X_seq = spline(i_seq, x_seq, I_seq);
Y_seq = spline(i_seq, y_seq, I_seq);
Z_seq = spline(i_seq, z_seq, I_seq);
path  = [X_seq', Y_seq', Z_seq'];

% 2) 路径长度
d    = diff(path,1,1);
Vc   = sum( sqrt(sum(d.^2,2)) );

% 3) 长方体碰撞检测（点落入任一AABB）
if any_point_in_boxes(path, boxes)
    Tc = inf;           % 直接判无效路径；也可用大罚值
else
    Tc = 0;
end

% 4) 越界检测
if any( path(:,1) < 0 | path(:,1) > mapRange(1) | ...
        path(:,2) < 0 | path(:,2) > mapRange(2) | ...
        path(:,3) < 0 | path(:,3) > mapRange(3) )
    Ec = inf;
else
    Ec = 0;
end

% 5) 目标函数
fitness = Vc + Tc + Ec;
end

function flag = any_point_in_boxes(P, boxes)
% 任一点落入任一长方体 -> true
flag = false;
for j = 1:size(boxes,1)
    b = boxes(j,:);
    in =  (P(:,1) >= b(1)) & (P(:,1) <= b(1)+b(4)) & ...
          (P(:,2) >= b(2)) & (P(:,2) <= b(2)+b(5)) & ...
          (P(:,3) >= b(3)) & (P(:,3) <= b(3)+b(6));
    if any(in)
        flag = true; return;
    end
end
end
