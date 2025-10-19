close all
clear  
clc
warning off;

%% 三维路径规划模型定义
global startPos goalPos N
N = 2;                                                     %  待优化点的个数(可以修改)
startPos = [10, 10, 10];                                   %  起点(可以修改)
goalPos = [175, 175, 50];                                  %  终点(可以修改)
SearchAgents_no = 30;                                      %  种群大小(可以修改)
Function_name = 'F1';                                      %  F1:随机产生地图 F2：导入固定地图
Max_iteration = 200;                                       %  最大迭代次数(可以修改)

% 获取函数细节
[lb, ub, dim, fobj] = Get_Functions_details(Function_name);

% 强制确保边界适合航迹规划（非负坐标）
fprintf('原始问题边界: lb=[%s], ub=[%s]\n', num2str(lb), num2str(ub));

% 对于航迹规划，确保所有坐标都是非负的
lb = max(lb, 0);  % 下边界不能小于0
if any(ub <= lb)
    ub = max(ub, max(goalPos) + 50);  % 确保上边界合理
end

fprintf('修正后边界: lb=[%s], ub=[%s]\n', num2str(lb), num2str(ub));
fprintf('起点: [%s], 终点: [%s]\n', num2str(startPos), num2str(goalPos));

% 验证起点和终点是否在边界内
for i = 1:length(startPos)
    curr_lb = lb(min(i, length(lb)));
    curr_ub = ub(min(i, length(ub)));
    
    if startPos(i) < curr_lb || startPos(i) > curr_ub
        fprintf('警告: 起点维度%d超出边界[%.2f, %.2f], 值=%.2f\n', i, curr_lb, curr_ub, startPos(i));
    end
    
    if goalPos(i) < curr_lb || goalPos(i) > curr_ub
        fprintf('警告: 终点维度%d超出边界[%.2f, %.2f], 值=%.2f\n', i, curr_lb, curr_ub, goalPos(i));
    end
end

% 算法列表（使用改进的WOA）
AlgorithmName = {'MIDBO', 'DBO', 'WOA', 'GWO'};
addpath('./AlgorithmCode/');

bestFit = [];
data = struct();

for i = 1:size(AlgorithmName, 2)
    fprintf('\n=== 开始运行算法: %s ===\n', AlgorithmName{i});
    
    Algorithm = str2func(AlgorithmName{i});
    
    try
        [Best_score, Best_pos, Convergence_curve] = Algorithm(SearchAgents_no, Max_iteration, lb, ub, dim, fobj);
        
        % 验证结果是否包含负值
        if any(Best_pos < 0)
            fprintf('错误: 算法%s返回了负坐标: [%s]\n', AlgorithmName{i}, num2str(Best_pos));
            % 修正负值
            Best_pos = max(Best_pos, 0);
            fprintf('已修正为: [%s]\n', num2str(Best_pos));
        end
        
        % 保存结果
        data(i).Best_score = Best_score;
        data(i).Best_pos = Best_pos;
        data(i).Convergence_curve = Convergence_curve;
        bestFit = [bestFit data(i).Best_score];
        
        fprintf('算法%s完成 - 最优值: %.6f, 最优位置: [%s]\n', ...
                AlgorithmName{i}, Best_score, num2str(Best_pos, '%.2f '));
                
    catch ME
        fprintf('算法%s运行出错: %s\n', AlgorithmName{i}, ME.message);
        % 提供默认值
        data(i).Best_score = inf;
        data(i).Best_pos = zeros(1, dim);
        data(i).Convergence_curve = inf * ones(1, Max_iteration);
        bestFit = [bestFit inf];
    end
end

% 显示结果
fprintf('\n=== 最终结果对比 ===\n');
fprintf('bestFit: [%s]\n', num2str(bestFit, '%.6f '));

for i = 1:size(data, 2)
    if isfinite(data(i).Best_score)
        fprintf('算法 %s - 最优值: %.6f, 位置: [%s]\n', ...
                AlgorithmName{i}, data(i).Best_score, num2str(data(i).Best_pos, '%.2f '));
    else
        fprintf('算法 %s - 运行失败\n', AlgorithmName{i});
    end
end

% 保存数据
save data data

%% 绘制结果图
% 创建Picture文件夹
if ~exist('./Picture','dir')
    mkdir('./Picture');
end

% 直方图
figure 
valid_fit = bestFit(isfinite(bestFit));
valid_names = AlgorithmName(isfinite(bestFit));

if ~isempty(valid_fit)
    bar(valid_fit)
    ylabel('适应度');
    set(gca,'xtick', 1:length(valid_names));
    set(gca,'XTickLabel', valid_names);
    title('各算法性能对比');
    grid on;
else
    text(0.5, 0.5, '所有算法都失败了', 'HorizontalAlignment', 'center');
end
set(gcf,'color','w');
saveas(gcf,'./Picture/直方图.jpg');

% 收敛曲线
strColor = {'r-','g-','b-','k-','m-','c-','y-'};
figure
legend_entries = {};
plot_count = 0;

for i = 1:size(data, 2)
    if isfinite(data(i).Best_score) && all(isfinite(data(i).Convergence_curve))
        plot_count = plot_count + 1;
        plot(data(i).Convergence_curve, strColor{mod(i-1,length(strColor))+1}, 'linewidth', 1.5);
        hold on;
        legend_entries{plot_count} = AlgorithmName{i};
    end
end

if plot_count > 0
    xlabel('迭代次数');
    ylabel('无人机飞行路径长度');
    legend(legend_entries, 'Location', 'Best');
    title('算法收敛曲线对比');
    grid on;
else
    text(0.5, 0.5, '没有有效的收敛数据', 'HorizontalAlignment', 'center');
end
set(gcf,'color','w');
saveas(gcf,'./Picture/收敛曲线.jpg');

%% 显示三维图
try
    set(0,'DefaultFigureVisible','on');
    path_pts = plotFigure_rect(data, AlgorithmName, strColor);
    hFig3 = gcf;
    ax3 = gca;
    
    view(ax3, 3);
    axis(ax3, 'equal');
    drawnow; 
    shg;
    
    saveas(hFig3, './Picture/路径曲线（三维）.jpg');
    
    % 生成二维图
    hFig2 = figure('Visible','off','Name','二维快照','NumberTitle','off');
    ax2 = copyobj(ax3, hFig2);
    set(ax2, 'Units','normalized','Position',[0.13 0.11 0.775 0.815]);
    view(ax2, 2);
    axis(ax2, 'equal');
    drawnow;
    
    saveas(hFig2, './Picture/路径曲线（二维）.jpg');
    close(hFig2);
    
    figure(hFig3); 
    drawnow; 
    shg;
    
    % 保存路径数据
    save('path_data.mat','path_pts');
    
catch ME
    fprintf('绘制路径图时出错: %s\n', ME.message);
end

fprintf('\n程序执行完成！\n');