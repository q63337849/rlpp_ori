function [Leader_score,Leader_pos,Convergence_curve]=WOA(SearchAgents_no,Max_iter,lb,ub,dim,fobj)
%_________________________________________________________________________%
%  改进的Whale Optimization Algorithm (WOA) for 航迹规划                    %
%  完全解决负坐标问题的版本                                                   %
%_________________________________________________________________________%

% 强制确保边界为非负（航迹规划专用）
fprintf('原始边界 - lb: [%s], ub: [%s]\n', num2str(lb), num2str(ub));

% 对于航迹规划，强制设置最小边界为0
lb = max(lb, 0);  % 确保所有下边界都不小于0

% 如果上边界也小于0，则设置合理的默认值
if any(ub <= 0)
    ub = max(ub, 200);  % 设置合理的上边界
end

fprintf('修正后边界 - lb: [%s], ub: [%s]\n', num2str(lb), num2str(ub));

% 初始化领导者位置和得分
Leader_pos = zeros(1,dim);
Leader_score = inf;

% 安全初始化搜索代理位置
Positions = safe_initialization(SearchAgents_no, dim, ub, lb);

% 验证初始化结果
[min_vals, max_vals] = check_position_bounds(Positions, lb, ub);
fprintf('初始化后位置范围: 最小值[%s], 最大值[%s]\n', num2str(min_vals), num2str(max_vals));

Convergence_curve = zeros(1,Max_iter);
t = 0;

% 主循环
while t < Max_iter
    % 评估所有搜索代理
    for i = 1:size(Positions,1)
        % 严格边界检查和修正
        Positions(i,:) = strict_boundary_enforcement(Positions(i,:), ub, lb);
        
        % 计算适应度
        fitness = fobj(Positions(i,:));
        
        % 检查适应度是否有效
        if ~isfinite(fitness) || isnan(fitness)
            fprintf('警告: 代理%d的适应度无效，重新初始化\n', i);
            Positions(i,:) = safe_random_position(dim, ub, lb);
            fitness = fobj(Positions(i,:));
        end
        
        % 更新领导者
        if fitness < Leader_score
            Leader_score = fitness;
            Leader_pos = Positions(i,:);
        end
    end
    
    % WOA参数更新
    a = 2 - t * ((2)/Max_iter);  % a从2线性递减到0
    a2 = -1 + t * ((-1)/Max_iter);  % a2从-1线性递减到-2
    
    % 更新搜索代理位置
    for i = 1:size(Positions,1)
        % 保存当前位置作为备份
        backup_pos = Positions(i,:);
        
        % 为每个维度更新位置
        for j = 1:dim
            % 获取当前维度边界
            curr_lb = get_boundary_value(lb, j);
            curr_ub = get_boundary_value(ub, j);
            
            % 生成随机参数
            r1 = rand();
            r2 = rand();
            p = rand();
            
            A = 2*a*r1 - a;  % 控制参数A
            C = 2*r2;        % 控制参数C
            
            % 位置更新
            if p < 0.5
                if abs(A) >= 1
                    % 全局搜索阶段
                    new_pos = global_search_update(Positions, i, j, A, C, curr_lb, curr_ub, SearchAgents_no);
                else
                    % 局部开发阶段
                    new_pos = local_exploitation_update(Positions(i,j), Leader_pos(j), A, C, curr_lb, curr_ub);
                end
            else
                % 螺旋更新
                new_pos = spiral_update(Positions(i,j), Leader_pos(j), a2, curr_lb, curr_ub);
            end
            
            % 应用新位置
            Positions(i,j) = new_pos;
        end
        
        % 最终边界检查
        Positions(i,:) = strict_boundary_enforcement(Positions(i,:), ub, lb);
        
        % 验证更新结果
        if any(Positions(i,:) < 0) || any(isnan(Positions(i,:))) || any(~isfinite(Positions(i,:)))
            fprintf('严重错误: 代理%d位置无效 [%s]，恢复备份位置\n', i, num2str(Positions(i,:)));
            Positions(i,:) = backup_pos;
        end
    end
    
    t = t + 1;
    Convergence_curve(t) = Leader_score;
    
    % 定期验证和报告
    if mod(t, 50) == 0 || t == 1
        [min_vals, max_vals] = check_position_bounds(Positions, lb, ub);
        fprintf('迭代%d: 适应度=%.6f, 位置范围[%.2f,%.2f]\n', t, Leader_score, min(min_vals), max(max_vals));
        
        if any(min_vals < 0)
            fprintf('错误: 检测到负坐标! 最小值: [%s]\n', num2str(min_vals));
        end
    end
end

fprintf('最终结果: 最优适应度=%.6f, 最优位置=[%s]\n', Leader_score, num2str(Leader_pos));

end

%% 辅助函数

% 安全初始化函数
function Positions = safe_initialization(SearchAgents_no, dim, ub, lb)
    Positions = zeros(SearchAgents_no, dim);
    
    for i = 1:SearchAgents_no
        for j = 1:dim
            curr_lb = get_boundary_value(lb, j);
            curr_ub = get_boundary_value(ub, j);
            
            % 确保边界有效
            if curr_lb < 0
                curr_lb = 0;
            end
            if curr_ub <= curr_lb
                curr_ub = curr_lb + 100;
            end
            
            Positions(i,j) = curr_lb + rand() * (curr_ub - curr_lb);
        end
    end
end

% 获取边界值的辅助函数
function val = get_boundary_value(boundary, index)
    if length(boundary) == 1
        val = boundary;
    else
        val = boundary(min(index, length(boundary)));
    end
    % 确保非负
    val = max(0, val);
end

% 严格边界约束函数
function pos = strict_boundary_enforcement(pos, ub, lb)
    for j = 1:length(pos)
        curr_lb = get_boundary_value(lb, j);
        curr_ub = get_boundary_value(ub, j);
        
        % 处理无效值
        if ~isfinite(pos(j)) || isnan(pos(j))
            pos(j) = curr_lb + rand() * (curr_ub - curr_lb);
        else
            % 严格约束到边界内
            pos(j) = max(curr_lb, min(curr_ub, pos(j)));
        end
        
        % 双重检查确保非负
        pos(j) = max(0, pos(j));
    end
end

% 全局搜索更新
function new_pos = global_search_update(Positions, current_idx, dim_idx, A, C, lb, ub, SearchAgents_no)
    % 选择随机搜索代理
    rand_idx = randi(SearchAgents_no);
    while rand_idx == current_idx
        rand_idx = randi(SearchAgents_no);
    end
    
    X_rand = Positions(rand_idx, dim_idx);
    
    % 限制参数范围
    C_safe = max(0.1, min(2.0, abs(C)));
    A_safe = max(-1.5, min(1.5, A));
    
    % 计算距离和新位置
    D = abs(C_safe * X_rand - Positions(current_idx, dim_idx));
    new_pos = X_rand - A_safe * D;
    
    % 边界约束
    new_pos = max(lb, min(ub, new_pos));
end

% 局部开发更新
function new_pos = local_exploitation_update(current_pos, leader_pos, A, C, lb, ub)
    % 限制参数
    C_safe = max(0.1, min(2.0, abs(C)));
    A_safe = max(-1.0, min(1.0, A));
    
    % 计算新位置
    D = abs(C_safe * leader_pos - current_pos);
    new_pos = leader_pos - A_safe * D;
    
    % 边界约束
    new_pos = max(lb, min(ub, new_pos));
end

% 螺旋更新
function new_pos = spiral_update(current_pos, leader_pos, a2, lb, ub)
    % 螺旋参数
    b = 1;
    l = (a2 - 1) * rand + 1;
    l = max(-1, min(1, l));  % 限制l的范围
    
    % 计算距离
    distance = abs(leader_pos - current_pos);
    
    % 螺旋更新，限制最大步长
    max_step = (ub - lb) * 0.2;  % 最大步长为范围的20%
    spiral_component = distance * exp(b * l) * cos(l * 2 * pi);
    spiral_component = max(-max_step, min(max_step, spiral_component));
    
    new_pos = leader_pos + spiral_component;
    
    % 边界约束
    new_pos = max(lb, min(ub, new_pos));
end

% 生成安全的随机位置
function pos = safe_random_position(dim, ub, lb)
    pos = zeros(1, dim);
    for j = 1:dim
        curr_lb = get_boundary_value(lb, j);
        curr_ub = get_boundary_value(ub, j);
        pos(j) = curr_lb + rand() * (curr_ub - curr_lb);
    end
end

% 检查位置边界
function [min_vals, max_vals] = check_position_bounds(Positions, lb, ub)
    min_vals = min(Positions, [], 1);
    max_vals = max(Positions, [], 1);
    
    % 检查是否有越界
    for j = 1:size(Positions, 2)
        curr_lb = get_boundary_value(lb, j);
        curr_ub = get_boundary_value(ub, j);
        
        if min_vals(j) < curr_lb || max_vals(j) > curr_ub
            fprintf('维度%d越界: 范围[%.2f, %.2f], 边界[%.2f, %.2f]\n', ...
                    j, min_vals(j), max_vals(j), curr_lb, curr_ub);
        end
    end
end