
close all
clear  
clc
warning off;

%% 三维路径规划模型定义
global startPos goalPos N
N=2;                                                     %  待优化点的个数(可以修改)
startPos = [10, 10, 10];                                 %  起点(可以修改)
goalPos = [175, 175, 50];                                 %  终点(可以修改)
SearchAgents_no=30;                                      %  种群大小(可以修改)
Function_name='F1';                                      %  F1:随机产生地图 F2：导入固定地图
Max_iteration=200;                                       %  最大迭代次数(可以修改)
% Load details of the selected benchmark function
[lb,ub,dim,fobj]=Get_Functions_details(Function_name);
AlgorithmName={'MIDBO','DBO','WOA','GWO'};                           %  算法名称
addpath('./AlgorithmCode/');                             %  添加算法路径
bestFit=[];                                              %  保存各算法的最优适应度值
for i=1:size(AlgorithmName,2)                            %  遍历每个算法，依次求解当前问题
    Algorithm=str2func(AlgorithmName{i});                    %  获取当前算法名称，并将字符转换为函数
    [Best_score,Best_pos,Convergence_curve]=Algorithm(SearchAgents_no,Max_iteration,lb,ub,dim,fobj);%当前算法求解
    %将当前算法求解结果放入data中
    data(i).Best_score=Best_score;                           %  保存该算法的Best_score到data
    data(i).Best_pos=Best_pos;                               %  保存该算法的Best_pos到data
    data(i).Convergence_curve=Convergence_curve;             %  保存该算法的Convergence_curve到data
    bestFit=[bestFit data(i).Best_score];
end  

disp('bestFit:');
disp(bestFit);
for i=1:size(data,2)
    disp(['算法 ', AlgorithmName{i}, ' 最优值: ', num2str(data(i).Best_score)]);
end

save data data
%%  画各算法的直方图
figure 
bar(bestFit)
ylabel('适应度');
set(gca,'xtick',1:1:size(AlgorithmName,2));
set(gca,'XTickLabel',AlgorithmName)
set(gcf,'color','w')
saveas(gcf,'./Picture/直方图.jpg') %将图片保存到Picture文件夹下面

%%  画收敛曲线
strColor={'r-','g-','b-','k-','m-','c-','y-'};
figure
for i=1:size(data,2)
plot(data(i).Convergence_curve,strColor{i},'linewidth',1.5)%semilogy
hold on
end
xlabel('迭代次数');
ylabel('适应度');
legend(AlgorithmName,'Location','Best')
set(gcf,'color','w')
saveas(gcf,'./Picture/收敛曲线.jpg') %将图片保存到Picture文件夹下面

%% 显示三维图并保存
set(0,'DefaultFigureVisible','on');   % 确保允许显示图窗
path_pts = plotFigure_rect(data, AlgorithmName, strColor);  % 只返回路径
hFig3 = gcf;                 % 三维图窗句柄
ax3   = gca;                 % 三维坐标轴句柄

% 三维视角与美化（可选，plotFigure_rect里若已设置可省略）
view(ax3, 3);
axis(ax3, 'equal');
drawnow; shg;

% 保存三维图
if ~exist('./Picture','dir'); mkdir('./Picture'); end
saveas(hFig3, './Picture/路径曲线（三维）.jpg');

% 额外保存路径数据（避免与内置 path 冲突）
save('path_data.mat','path_pts');

%% 生成二维图（复制坐标轴到临时图窗，避免影响三维图）
hFig2 = figure('Visible','off','Name','二维快照','NumberTitle','off');  % 隐藏窗口
ax2   = copyobj(ax3, hFig2);                 % 复制三维轴到新图
set(ax2, 'Units','normalized','Position',[0.13 0.11 0.775 0.815]); % 填满
view(ax2, 2);                                % 改为二维视角
axis(ax2, 'equal');
drawnow;

% 保存二维图并关闭临时窗口
saveas(hFig2, './Picture/路径曲线（二维）.jpg');
close(hFig2);

% 把三维图重新置前显示（用户看到的仍是三维）
figure(hFig3); drawnow; shg;




