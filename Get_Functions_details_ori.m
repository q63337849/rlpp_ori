
function [lb,ub,dim,fobj] = Get_Functions_details(F)
global X Y Z N mapRange
mapRange = [100,100,250];              % 地图长、宽、高范围
switch F
    case 'F1'
        %% 随机产生地图
        [X,Y,Z] = CreatModel(mapRange);
        [~,~,z]=peaks(100);%产生山峰地图
        H=45*(abs(z)+abs(z'))/2;
        MAPSIZE_X = size(H,2); % x index: columns of H
        MAPSIZE_Y = size(H,1); % y index: rows of H
        [X,Y] = meshgrid(1:MAPSIZE_X,1:MAPSIZE_Y); % Create all (x,y) points to plot
        Z=Z+H;
        save Z Z 
        SaveMapAsImage(X, Y, Z, 'random_map.png');
    case 'F2'
        %% 导入固定地图
        load('X.mat');
        load('Y.mat');
        load('Z.mat');
end
dim=3*N;
lb=0*ones(1,dim);%下限
ub=ones(1,dim);
ub(1:N)=mapRange(1);%X坐标上限
ub(1+N:2*N)=mapRange(2);%Y坐标上限
ub(1+2*N:3*N)=mapRange(3);%Z坐标上限
fobj = @Cost;%目标函数
end

function SaveMapAsImage(X, Y, Z, filename)
    % 绘制3D地图
    figure; % 创建新图窗
    surf(X, Y, Z); % 绘制3D曲面图
    shading flat; % 去除网格线，采用平坦着色
    title('随机生成的3D地图');
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    
    % 调整视角
    view(3); % 3D视角
    grid on; % 显示网格

    % 保存为图片文件
    saveas(gcf,'./Picture/地图（三维）.jpg'); % 保存当前图窗为文件，支持多种格式
end
