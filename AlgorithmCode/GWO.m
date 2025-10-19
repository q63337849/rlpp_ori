function [fMin, bestX, Convergence_curve] = GWO(pop, M, c, d, dim, fobj)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
lb = c .* ones(1, dim);    % Lower limit/bounds/ a vector
ub = d .* ones(1, dim);    % Upper limit/bounds/ a vector

% Initialization
for i = 1 : pop
    x(i, :) = lb + (ub - lb) .* rand(1, dim);
    fit(i) = fobj(x(i, :));
end

pFit = fit;
pX = x;

% Initialize alpha, beta, and delta positions
Alpha_pos = zeros(1, dim);
Alpha_score = inf; % Change this to -inf for maximization problems

Beta_pos = zeros(1, dim);
Beta_score = inf; % Change this to -inf for maximization problems

Delta_pos = zeros(1, dim);
Delta_score = inf; % Change this to -inf for maximization problems

% Find the best, second best, and third best solutions
[sorted_fit, sorted_idx] = sort(fit);
fMin = sorted_fit(1);      % fMin denotes the global optimum fitness value
bestX = x(sorted_idx(1), :);           % bestX denotes the global optimum position corresponding to fMin

% Initialize Alpha (best solution)
Alpha_score = sorted_fit(1);
Alpha_pos = x(sorted_idx(1), :);

% Initialize Beta (second best solution)
if pop >= 2
    Beta_score = sorted_fit(2);
    Beta_pos = x(sorted_idx(2), :);
else
    % If population size is 1, set Beta same as Alpha
    Beta_score = Alpha_score;
    Beta_pos = Alpha_pos;
end

% Initialize Delta (third best solution)  
if pop >= 3
    Delta_score = sorted_fit(3);
    Delta_pos = x(sorted_idx(3), :);
else
    % If population size is less than 3, set Delta same as Beta
    Delta_score = Beta_score;
    Delta_pos = Beta_pos;
end

% Main loop
for t = 1 : M
    
    for i = 1 : pop
        
        % Update Alpha, Beta, and Delta
        if pFit(i) < Alpha_score 
            % Shift the hierarchy down
            Delta_score = Beta_score;
            Delta_pos = Beta_pos;
            
            Beta_score = Alpha_score;
            Beta_pos = Alpha_pos;
            
            Alpha_score = pFit(i); % Update alpha
            Alpha_pos = pX(i, :);
        elseif pFit(i) < Beta_score 
            % Shift Delta down
            Delta_score = Beta_score;
            Delta_pos = Beta_pos;
            
            Beta_score = pFit(i); % Update beta
            Beta_pos = pX(i, :);
        elseif pFit(i) < Delta_score 
            Delta_score = pFit(i); % Update delta
            Delta_pos = pX(i, :);
        end
        
    end
    
    a = 2 - t * ((2) / M); % a decreases linearly from 2 to 0
    
    % Update the Position of search agents including omegas
    for i = 1 : pop
        for j = 1 : dim
            
            r1 = rand(); % r1 is a random number in [0,1]
            r2 = rand(); % r2 is a random number in [0,1]
            
            A1 = 2 * a * r1 - a; % Equation (3.3)
            C1 = 2 * r2;         % Equation (3.4)
            
            D_alpha = abs(C1 * Alpha_pos(j) - pX(i, j)); % Equation (3.5)-part 1
            X1 = Alpha_pos(j) - A1 * D_alpha; % Equation (3.6)-part 1
            
            r1 = rand();
            r2 = rand();
            
            A2 = 2 * a * r1 - a; % Equation (3.3)
            C2 = 2 * r2;         % Equation (3.4)
            
            D_beta = abs(C2 * Beta_pos(j) - pX(i, j)); % Equation (3.5)-part 2
            X2 = Beta_pos(j) - A2 * D_beta; % Equation (3.6)-part 2
            
            r1 = rand();
            r2 = rand(); 
            
            A3 = 2 * a * r1 - a; % Equation (3.3)
            C3 = 2 * r2;         % Equation (3.4)
            
            D_delta = abs(C3 * Delta_pos(j) - pX(i, j)); % Equation (3.5)-part 3
            X3 = Delta_pos(j) - A3 * D_delta; % Equation (3.6)-part 3
            
            x(i, j) = (X1 + X2 + X3) / 3; % Equation (3.7)
            
        end
        
        % Apply bounds
        x(i, :) = Bounds(x(i, :), lb, ub);
        
        % Calculate objective function for each search agent
        fit(i) = fobj(x(i, :));
        
    end
    
    % Update the individual's best fitness value and the global best fitness value
    for i = 1 : pop
        if (fit(i) < pFit(i))
            pFit(i) = fit(i);
            pX(i, :) = x(i, :);
        end

        if(pFit(i) < fMin)
            fMin = pFit(i);
            bestX = pX(i, :);
        end
    end
    
    Convergence_curve(t) = fMin;
end

% Application of simple limits/bounds
function s = Bounds(s, Lb, Ub)
% Apply the lower bound vector
temp = s;
I = temp < Lb;
temp(I) = Lb(I);

% Apply the upper bound vector
J = temp > Ub;
temp(J) = Ub(J);
% Update this new move
s = temp;