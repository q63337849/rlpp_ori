function [fMin, bestX, Convergence_curve] = MIDBO(pop, M, c, d, dim, fobj)
    % 改进版 MIDBO：混合引导 + 提前自适应变异 + Lévy 扰动 + 退化修复
    P_percent = 0.2;
    pNum = round(pop * P_percent);
    lb = c .* ones(1, dim);
    ub = d .* ones(1, dim);

    % 初始化
    x = lb + (ub - lb) .* rand(pop, dim);
    fit = zeros(1, pop);
    for i = 1:pop
        fit(i) = fobj(x(i, :));
    end
    pFit = fit;    % 个体历史最好
    pX = x;        % 个体历史最好位置
    [fMin, bestI] = min(fit);
    bestX = x(bestI, :);

    % 预分配收敛曲线
    Convergence_curve = inf(1, M);
    Convergence_curve(1) = fMin;

    % 主循环
    for t = 1:M
        % ====== 多精英引导机制 ======
        n_elite = 3;
        [~, idx_sort] = sort(pFit);
        eliteX = pX(idx_sort(1:n_elite), :);
        elite_mean_base = mean(eliteX, 1);
        [~, worst_idx] = max(pFit);
        worse = pX(worst_idx, :);

        % 混合引导：early stage 更依赖 global best，后期融合 elite_mean
        w = max(0.7 - 0.5 * (t / M), 0.3);  % 从 ~0.7 下降到 0.3
        guidance = w * bestX + (1 - w) * elite_mean_base;

        % Producers 更新（前 pNum 个）
        r2 = rand;
        for i = 1:pNum
            if r2 < 0.9
                a = 2 * (rand > 0.1) - 1; % 1 或 -1 
                x(i, :) = pX(i, :) + 0.3 * abs(pX(i, :) - worse) + a * 0.1 * pX(i, :);
            else
                theta = randperm(180, 1) * pi / 180;
                x(i, :) = pX(i, :) + tan(theta) .* abs(pX(i, :) - pX(i, :));
            end
            x(i, :) = Bounds(x(i, :), lb, ub);
            fit(i) = fobj(x(i, :));
        end

        % 多精英引导分组更新（其余个体）
        R = 1 - t / M;
        Xnew1 = Bounds(guidance .* (1 - R), lb, ub);
        Xnew2 = Bounds(guidance .* (1 + R), lb, ub);
        for i = (pNum + 1):pop
            weight = (i - 1) / (pop - 1);
            x(i, :) = weight * guidance + ...
                      ((rand(1, dim)) .* (pX(i, :) - Xnew1) + (rand(1, dim)) .* (pX(i, :) - Xnew2));
            x(i, :) = Bounds(x(i, :), Xnew1, Xnew2);
            fit(i) = fobj(x(i, :));
        end

        % 个体历史更新
        for i = 1:pop
            if fit(i) < pFit(i)
                pFit(i) = fit(i);
                pX(i, :) = x(i, :);
            end
            if pFit(i) < fMin
                fMin = pFit(i);
                bestX = pX(i, :);
            end
        end

        % =========== Lévy 大步扰动 ================
        % 早期增强跳跃，后期递增（可调整）
        if t <= round(0.2 * M)
            levy_prob = 0.7;
            levy_scale = 0.25;
        else
            levy_prob = 0.3 + 0.4 * (t / M);
            levy_scale = 0.14 + 0.12 * (t / M);
        end
        beta = 1.5;
        sigma_levy = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
        for i = 1:pop
            if rand < levy_prob
                u = randn(1,dim) * sigma_levy;
                v = randn(1,dim);
                step = u ./ abs(v).^(1/beta);
                x_levy = pX(i,:) + levy_scale * step .* (ub - lb);
                x_levy = Bounds(x_levy, lb, ub);
                fit_levy = fobj(x_levy);
                if fit_levy < pFit(i)
                    pFit(i) = fit_levy;
                    pX(i,:) = x_levy;
                end
                if fit_levy < fMin
                    fMin = fit_levy;
                    bestX = x_levy;
                end
            end
        end

        % ========== 提前激活混合变异（轻扰动 + 停滞放大） ==========
        % 轻扰动每代小概率发生
        for i = 1:pop
            if rand < 0.1
                x_mut = pX(i,:) + 0.01 * randn(1,dim) .* pX(i,:);
                x_mut = Bounds(x_mut, lb, ub);
                fit_mut = fobj(x_mut);
                if fit_mut < pFit(i)
                    pFit(i) = fit_mut;
                    pX(i,:) = x_mut;
                end
                if fit_mut < fMin
                    fMin = fit_mut;
                    bestX = x_mut;
                end
            end
        end

        % 检查最近 stagnation（过去几个代几乎无提升）
        stagnation_window = 3;
        if t > stagnation_window
            recent_improvements = abs(Convergence_curve(max(1, t - (1:stagnation_window))) - fMin);
            if all(recent_improvements < 1e-8)
                for i = 1:pop
                    if rand < 0.7
                        x_mut = pX(i,:) + randn(1,dim) .* pX(i,:);  % 强变异
                    else
                        x_mut = lb + (ub - lb) .* rand(1,dim);     % 随机重启
                    end
                    x_mut = Bounds(x_mut, lb, ub);
                    fit_mut = fobj(x_mut);
                    if fit_mut < pFit(i)
                        pFit(i) = fit_mut;
                        pX(i,:) = x_mut;
                    end
                    if fit_mut < fMin
                        fMin = fit_mut;
                        bestX = x_mut;
                    end
                end
            end
        end

        % =========== 天敌预警扰动机制 ==============
        danger_k = 5; danger_var_th = 1e-6; danger_ratio = 0.2;
        if t > danger_k && (fMin == Convergence_curve(max(1, t-danger_k)) || var(pFit) < danger_var_th)
            num_danger = max(1, round(danger_ratio * pop));
            [~, worst_idx4] = maxk(pFit, num_danger);
            for ii = 1:length(worst_idx4)
                pX(worst_idx4(ii), :) = lb + (ub - lb).*rand(1,dim);
                pFit(worst_idx4(ii)) = fobj(pX(worst_idx4(ii), :));
                if pFit(worst_idx4(ii)) < fMin
                    fMin = pFit(worst_idx4(ii));
                    bestX = pX(worst_idx4(ii), :);
                end
            end
        end

        % ========== 记录当前最优 ==========
        Convergence_curve(t) = fMin;
    end
end

%% 辅助：边界处理
function s = Bounds(s, Lb, Ub)
    s = min(max(s, Lb), Ub);
end
