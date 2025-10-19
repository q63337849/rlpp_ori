function [lb,ub,dim,fobj] = Get_Functions_details(F)
global N mapRange boxes

mapRange = [200,200,200];   % 地图长、宽、高

switch F
    case 'F1'  % 随机产生长方体障碍环境
        K = 20;                % 障碍数量（可调）
        minSize = [8, 8, 40];  % 每个维度的最小尺寸
        maxSize = [18,18,120]; % 每个维度的最大尺寸
        minGap  = 6;           % 障碍-障碍/障碍-边界 间隙
        boxes = gen_rect_obstacles(K, mapRange, minSize, maxSize, minGap);

    case 'F2'  % 固定参数（13个障碍）
        boxes = [...
            15  20   0   10 12 60;
            35  25   0   12 10 80;
            55  30   0   14 10 90;
            75  20   0   10 14 70;
            20  55   0   12 12 85;
            45  65   0   16 10 60;
            70  55   0   12 16 95;
            85  40   0   10 10 50;
            30  80   0   14 12 70;
            55  85   0   12 14 80;
            80  75   0   10 12 65;
            10  35   0   10 10 55;
            90  60   0   10 10 60  ];
    otherwise
        error('未知场景 F')
end

dim = 3*N;
lb  = zeros(1,dim);
ub  = ones(1,dim);
ub(1:N)           = mapRange(1);   % X
ub(N+1:2*N)       = mapRange(2);   % Y
ub(2*N+1:3*N)     = mapRange(3);   % Z

fobj = @Cost_SPSO_rect;                  % 使用新的代价函数Cost_SPSO_rect/CostRect
end
