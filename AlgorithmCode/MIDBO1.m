
function [fMin , bestX, Convergence_curve] = MIDBO(pop, M,c,d,dim,fobj  )

P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers
lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector
%Initialization
for i = 1 : pop

    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );
    fit( i ) = fobj( x( i, : ) ) ;
end
pFit = fit;
pX = x;
XX=pX;
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
% Start updating the solutions.
for t = 1 : M
    t
    [~,idx]=sort(pFit);
    [fmax,B]=max(pFit);
    worse= pX(B,:);
    r2=rand(1);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : pNum
        if(r2<0.9)
            r1=rand(1);
            a=rand(1,1);
            if (a>0.1)
                a=1;
            else
                a=-1;
            end
            x( i , : ) =  pX(  i , :)+0.3*abs(pX(i , : )-worse)+a*0.1*(XX( i , :)); % Equation (1)
        else

            aaa= randperm(180,1);
            if ( aaa==0 ||aaa==90 ||aaa==180 )
                x(  i , : ) = pX(  i , :);
            end
            theta= aaa*pi/180;

            x(  i , : ) = pX(  i , :)+tan(theta).*abs(pX(i , : )-XX( i , :));    % Equation (2)
        end

        x(  i , : ) = Bounds( x(i , : ), lb, ub );
        fit(  i  ) = fobj( x(i , : ) );
    end
    [ fMMin, bestII ] = min( fit );      % fMin denotes the current optimum fitness value
    bestXX = x( bestII, : );             % bestXX denotes the current optimum position
    R=1-t/M;                           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xnew1 = bestXX.*(1-R);
    Xnew2 =bestXX.*(1+R);                    %%% Equation (3)
    Xnew1= Bounds( Xnew1, lb, ub );
    Xnew2 = Bounds( Xnew2, lb, ub );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Xnew11 = bestX.*(1-R);
    Xnew22 =bestX.*(1+R);                     %%% Equation (5)
    Xnew11= Bounds( Xnew11, lb, ub );
    Xnew22 = Bounds( Xnew22, lb, ub );
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = ( pNum + 1 ) :12                  % Equation (4)
        %% 改进点1：改进雏球和偷窃蜣螂对最优解的接受程度
        x( i, : )=(idx(i)-1)/(pop-1)* bestXX+((rand(1,dim)).*(pX( i , : )-Xnew1)+(rand(1,dim)).*(pX( i , : )-Xnew2));
        x(i, : ) = Bounds( x(i, : ), Xnew1, Xnew2 );
        fit(i ) = fobj(  x(i,:) ) ;
    end

    for i = 13: 19                  % Equation (6)

        x( i, : )=pX( i , : )+((randn(1)).*(pX( i , : )-Xnew11)+((rand(1,dim)).*(pX( i , : )-Xnew22)));
        x(i, : ) = Bounds( x(i, : ),lb, ub);
        fit(i ) = fobj(  x(i,:) ) ;

    end

    for j = 20 : pop                 % Equation (7)
        %% 改进点1：改进雏球和偷窃蜣螂对最优解的接受程度
        x( j,: )=(idx(j)-1)/(pop-1)*bestX+randn(1,dim).*((abs(( pX(j,:  )-bestXX)))+(abs(( pX(j,:  )-bestX))))./2;
        x(j, : ) = Bounds( x(j, : ), lb, ub );
        fit(j ) = fobj(  x(j,:) ) ;
    end
    % Update the individual's best fitness vlaue and the global best fitness value
    XX=pX;
    for i = 1 : pop
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end

        if( pFit( i ) < fMin )
            % fMin= pFit( i );
            fMin= pFit( i );
            bestX = pX( i, : );
        end
    end

    %% 改进点2：融合麻雀搜索算法追随机制的扰动策略
    [~,w_idx]=sort(pFit);
    worseX = pX (w_idx(end),:);
    newbestXX = pX (w_idx(1),:);
    pv =    2*(1-(t/M)^1.5)/3;
    for i = 1 : pop
        if rand < pv
            A=floor(rand(1,dim)*2)*2-1;
            rr1 = rand;
            ST = 0.7;
            if rr1>ST
                newx(i,:)=randn(1)*exp((worseX-pX(i, : ))/(i)^2);
            else
                newx(i,:)=newbestXX+(abs(( pX(i,:)-newbestXX)))*(A'*(A*A')^(-1))*ones(1,dim);
            end
            newx(i,:) = Bounds(newx(i,:),lb, ub);
            newfit = fobj(newx(i,:));
            if newfit< pFit( i )
                pFit( i ) = newfit;
                pX( i, : ) = newx( i, : );
            end
            if(newfit < fMin )
                fMin= newfit;
                bestX = newx(i,:);
            end
        end
    end

    Convergence_curve(t)=fMin;

    %% 改进点3：柯西高斯变异
    % ==== 混合变异改进 with 自适应概率 ====
p = min(0.1 + 0.9 * t / M,1); % 当前迭代自适应概率

if t>5 && Convergence_curve(t-5)==Convergence_curve(t)
    for i = 1:pop
        if rand < p
            % 高斯变异（后期概率大）
            x_mut = pX(i,:) + randn(1,dim) .* pX(i,:);
        else
            % 柯西变异（前期概率大）
            x_mut = pX(i,:) + cauchyrnd(0,1,[1,dim]) .* pX(i,:);
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

% ==== 天敌预警扰动机制 ====
% 参数设定
danger_k = 5;                % 连续几代未进步
danger_var_th = 1e-6;        % 群体方差触发阈值
danger_ratio = 0.2;          % 参与扰动的最差个体比例（20%）

% 触发条件：多代收敛停滞 或 群体适应度方差过小
danger = false;
if t > danger_k && (Convergence_curve(t) == Convergence_curve(t-danger_k) || var(pFit)<danger_var_th)
    danger = true;
end

if danger
    num_danger = max(1, round(danger_ratio * pop));
    [~, worst_idx] = maxk(pFit, num_danger); % 找到适应度最差的num_danger个体索引
    for ii = 1:length(worst_idx)
        if rand < 0.5
            % 方式一：完全跳到新随机解
            x(worst_idx(ii), :) = lb + (ub-lb) .* rand(1, dim);
        else
            % 方式二：跳到当前最优解远处的大扰动
            scale = 1 + rand(); % 可自定义扰动强度
            x(worst_idx(ii), :) = bestX + scale * randn(1,dim) .* (ub-lb);
            x(worst_idx(ii), :) = Bounds(x(worst_idx(ii),:), lb, ub); % 保证在范围内
        end
        fit(worst_idx(ii)) = fobj(x(worst_idx(ii), :));
    end
    % 更新个体最优
    for ii = 1:length(worst_idx)
        if fit(worst_idx(ii)) < pFit(worst_idx(ii))
            pFit(worst_idx(ii)) = fit(worst_idx(ii));
            pX(worst_idx(ii), :) = x(worst_idx(ii), :);
        end
        if fit(worst_idx(ii)) < fMin
            fMin = fit(worst_idx(ii));
            bestX = x(worst_idx(ii), :);
        end
    end
end





end
% Application of simple limits/bounds

function s = Bounds( s, Lb, Ub)
% Apply the lower bound vector
temp = s;
I = temp < Lb;
temp(I) = Lb(I);

% Apply the upper bound vector
J = temp > Ub;
temp(J) = Ub(J);
% Update this new move
s = temp;
function S = Boundss( SS, LLb, UUb)
% Apply the lower bound vector
temp = SS;
I = temp < LLb;
temp(I) = LLb(I);

% Apply the upper bound vector
J = temp > UUb;
temp(J) = UUb(J);
% Update this new move
S = temp;

