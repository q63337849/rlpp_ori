
function fitness = Cost(x)
global N startPos goalPos X Y Z mapRange
% 利用三次样条拟合散点
x_seq=[startPos(1), x(1:N), goalPos(1)];
y_seq=[startPos(2), x(N+1:2*N), goalPos(2)];
z_seq=[startPos(3), x(2*N+1:3*N), goalPos(3)];
k = length(x_seq);
i_seq = linspace(0,1,k);
I_seq = linspace(0,1,80);
X_seq = spline(i_seq,x_seq,I_seq);
Y_seq = spline(i_seq,y_seq,I_seq);
Z_seq = spline(i_seq,z_seq,I_seq);
path = [X_seq', Y_seq', Z_seq'];%生成路径

%% 计算三次样条得到的离散点的路径长度（适应度）
dx = diff(X_seq);
dy = diff(Y_seq);
dz = diff(Z_seq);
Vc = sum(sqrt(dx.^2 + dy.^2 + dz.^2));

%% 判断生成的曲线是否与与障碍物相交
Tc = 0;
for i = 2:size(path,1)
    x = path(i,1);
    y = path(i,2);
    z_interp = interp2(X,Y,Z,x,y);
    if path(i,3) < z_interp
        Tc = inf;
        break
    end
end

%% 判断生成的曲线是否在了指定空域内
Ec = 0;
for i = 1:size(path,1)
    if sum( (path(i,:)<=mapRange)&(path(i,:)>=0))==3
        Ec = 0;
    else
        Ec = inf;
         break
    end       
end

%% 目标函数
fitness=Vc+Tc+Ec;%无人机的飞行路径的目标函数主要由三部分组成，分别是无人机总飞行航程、绕过障碍物的代价和在规定边界内飞行的代价。（来自公众号：强盛机器学习）

end