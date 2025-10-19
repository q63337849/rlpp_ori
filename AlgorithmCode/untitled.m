% 蜣螂优化算法生成初始种群的Matlab代码
% 假设优化问题在二维搜索空间内进行

% 参数设置
population_size = 100;  % 初始种群大小
dimensions = 2;        % 问题的维度
upper_bound = 10;      % 搜索空间的上界
lower_bound = -10;     % 搜索空间的下界

% 生成初始种群
initial_population = lower_bound + (upper_bound - lower_bound) * rand(population_size, dimensions);

% 绘制初始种群分布图
figure;
scatter(initial_population(:, 1), initial_population(:, 2), 'filled');
title('蜣螂优化算法生成的初始种群分布');
xlabel('维度1');
ylabel('维度2');
xlim([lower_bound, upper_bound]);
ylim([lower_bound, upper_bound]);
grid on;

% 显示图形
legend('初始种群个体');
