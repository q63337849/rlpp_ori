% 蜣螂优化算法 - 使用Chebyshev混沌映射生成初始种群
population_size = 100; % 初始种群大小
dimensions = 2; % 搜索空间维度
lower_bound = -10; % 搜索空间下边界
upper_bound = 10; % 搜索空间上边界

% Chebyshev混沌映射参数
iterations = 2000; % 迭代次数
k = 50; % Chebyshev映射参数
x = zeros(iterations, 1);
x(1) = rand; % 初始值
for i = 2:iterations
    x(i) = cos(k * acos(x(i-1))); % Chebyshev混沌映射
end

% 使用混沌序列生成初始种群
initial_population = zeros(population_size, dimensions);
for i = 1:population_size
    for j = 1:dimensions
        idx = mod(i + j - 2, iterations) + 1;
        initial_population(i, j) = lower_bound + (upper_bound - lower_bound) * (x(idx) + 1) / 2;
    end
end

disp('使用Chebyshev混沌映射生成的初始种群:');
disp(initial_population);

% 绘制初始种群图
figure;
scatter(initial_population(:, 1), initial_population(:, 2), 'filled');
xlabel('Dimension 1');
ylabel('Dimension 2');
title('使用Chebyshev混沌映射生成的初始种群分布');
grid on;
