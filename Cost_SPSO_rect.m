function fitness = Cost_SPSO_rect(x)
% 总体代价函数 F = sum_{k=1..4} b_k * F_k  (式(9))
% 权重按你的要求: [5 1 10 1] —— 路径长度、威胁、安全高度、平滑度
% 适配你当前的样条路径/长方体障碍环境

% 需要的全局量
global N startPos goalPos mapRange boxes

% ========= 可调参数（请按需要修改/外置到配置） =========
bk   = [5, 1, 10, 1];     % b1,b2,b3,b4  —— 你要求的 5 1 10 1
D    = 1.0;               % UAV 直径（或安全半径*2）；影响威胁硬碰撞阈值
S    = 8.0;               % 危险缓冲距离（线性罚区宽度）
hmin = 5;                % 允许的最低相对高度（对平地为 z）
hmax = 100;               % 允许的最高相对高度
a1   = 1.0;               % F4 里转弯角权重（式(8)的 a1）
a2   = 1.0;               % F4 里爬升角变化量权重（式(8)的 a2）
nsample = 120;            % 路径采样点数（越大越“严”）
% =====================================================

% 1) 样条生成路径离散点（与现有流程一致）
x_seq = [startPos(1), x(1:N),             goalPos(1)];
y_seq = [startPos(2), x(N+1:2*N),         goalPos(2)];
z_seq = [startPos(3), x(2*N+1:3*N),       goalPos(3)];
k     = numel(x_seq);
i_seq = linspace(0,1,k);
I_seq = linspace(0,1,nsample);
X_seq = spline(i_seq, x_seq, I_seq);
Y_seq = spline(i_seq, y_seq, I_seq);
Z_seq = spline(i_seq, z_seq, I_seq);
P     = [X_seq(:), Y_seq(:), Z_seq(:)];      % n x 3

% 2) 边界越界 -> 直接无穷大（排除不可行解）
if any(P(:,1) < 0 | P(:,1) > mapRange(1) | ...
       P(:,2) < 0 | P(:,2) > mapRange(2) | ...
       P(:,3) < 0 | P(:,3) > mapRange(3))
    fitness = inf; return;
end

% ========== F1: 路径长度 (式(1)) ==========
dP = diff(P,1,1);
segLen = sqrt(sum(dP.^2,2));
F1 = sum(segLen);   % 【公式(1)】

% ========== F2: 威胁代价 (式(2)思想 → AABB等效) ==========
% 对每一段与每个长方体，计算：若段穿入“硬碰撞盒”(AABB 膨胀 D) => inf
% 否则取段到“硬碰撞盒”的最小距离 dk；若 dk < S 则线性罚 (S - dk)
% 说明：原文以“圆柱 + (S, D, Rk)”；此处将 Rk 合入盒的半径等效为 AABB 膨胀（Rk=0）
F2 = 0; 
for j = 1:size(P,1)-1
    p0 = P(j,:); p1 = P(j+1,:);
    for kbox = 1:size(boxes,1)
        b = boxes(kbox,:); % [x y z w l h]
        hardBox = inflateAABB(b, D);  % 硬碰撞区：AABB 外扩 D
        % a) 硬碰撞判定：线段是否与 hardBox 相交
        if segmentAABBIntersect(p0,p1, hardBox)
            fitness = inf; return; % Tk = inf
        end
        % b) 线性罚区：到 hardBox 的最小距离 dk
        dk = segmentAABBDistance(p0,p1, hardBox);
        if dk < S
            F2 = F2 + (S - dk);     % Tk = S - dk
        end
    end
end

% ========== F3: 安全高度 (式(3)(4)) ==========
% 平地：相对高度 hij = z；如有地形可改为 hij = z - terrain(x,y)
hij = Z_seq(:); 
Hij = zeros(size(hij));
inRange = (hij >= hmin) & (hij <= hmax);
Hij(inRange)  = abs(hij(inRange) - (hmax+hmin)/2);   % |h - (hmax+hmin)/2|
Hij(~inRange) = inf;                                  % 超界直接不可行
F3 = sum(Hij);  % 【式(4)】

if isinf(F3)  % 提前剪枝
    fitness = inf; return;
end

% ========== F4: 平滑度 (式(8): 转弯角 + 爬升角变化) ==========
% 用样条采样点近似：投影到平面求转弯角φ，三维段求爬升角ψ
phi = zeros(max(0,size(P,1)-2),1);
psi = zeros(size(P,1)-1,1);

% 爬升角 ψ：与水平投影的夹角
for j = 1:size(P,1)-1
    v   = P(j+1,:)-P(j,:);
    vp  = [v(1), v(2), 0];           % 水平投影
    num = v(3);
    den = norm(vp);
    if den==0, psi(j)=sign(num)*pi/2; else, psi(j)=atan(num/den); end
end

% 转弯角 φ：相邻两段的水平投影夹角
for j = 1:size(P,1)-2
    v1p = [P(j+1,1)-P(j,1),   P(j+1,2)-P(j,2), 0];
    v2p = [P(j+2,1)-P(j+1,1), P(j+2,2)-P(j+1,2), 0];
    nrm = norm(v1p)*norm(v2p);
    if nrm==0
        phi(j) = 0;
    else
        crossz = v1p(1)*v2p(2) - v1p(2)*v2p(1);
        dotp   = v1p(1)*v2p(1) + v1p(2)*v2p(2);
        phi(j) = atan2(abs(crossz), dotp);   % 等价于式(6)的 arctan(||×|| / ·)
    end
end

F4 = a1*sum(phi) + a2*sum(abs(diff(psi)));  % 【式(8)】

% ========== 总体代价 (式(9)) ==========
fitness = bk(1)*F1 + bk(2)*F2 + bk(3)*F3 + bk(4)*F4;  % 【式(9)】fitness = bk(1)*F1 + bk(2)*F2 + bk(3)*F3 + bk(4)*F4

end

% ======== 几何子函数：AABB 膨胀 / 线段-盒 相交 / 距离 ========

function bout = inflateAABB(b, r)
% b = [x y z w l h]; 以各向同性 r 向外膨胀
bout = [b(1)-r, b(2)-r, b(3)-r, b(4)+2*r, b(5)+2*r, b(6)+2*r];
end

function tf = segmentAABBIntersect(p0,p1,b)
% slab 法线段-AABB 相交
bmin = b(1:3);
bmax = b(1:3) + b(4:6);
d = p1 - p0;
t0 = 0; t1 = 1;
for i=1:3
    if abs(d(i)) < 1e-12
        if p0(i) < bmin(i) || p0(i) > bmax(i), tf=false; return; end
    else
        invD = 1/d(i);
        tNear = (bmin(i)-p0(i))*invD;
        tFar  = (bmax(i)-p0(i))*invD;
        if tNear > tFar, tmp=tNear; tNear=tFar; tFar=tmp; end
        t0 = max(t0, tNear);
        t1 = min(t1, tFar);
        if t0 > t1, tf=false; return; end
    end
end
tf = true;
end

function d = segmentAABBDistance(p0,p1,b)
% 线段到 AABB 的最小距离（欧氏）
bmin = b(1:3); bmax = b(1:3)+b(4:6);
% 最近点投影：先求线段最近点到盒的 clamped 距离
% 将问题转为线段参数 t ∈ [0,1] 的 3D 有界最小化，这里用采样近似足够稳定
T = linspace(0,1,20);
dmin = inf;
for t = T
    p = p0 + t*(p1-p0);
    q = min(max(p, bmin), bmax);   % p clamp 到盒面
    dmin = min(dmin, norm(p - q));
end
d = dmin;
end
