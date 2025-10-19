
function path=plotFigure(data,LengendStr,strcolor)
global N startPos goalPos X Y Z 
num=size(data,2);
for i=1:num %每个无人机
    x=data(i).Best_pos;
    x_seq=[startPos(1), x(1:N), goalPos(1)];
    y_seq=[startPos(2), x(N+1:2*N), goalPos(2)];
    z_seq=[startPos(3), x(2*N+1:3*N), goalPos(3)];
    k = length(x_seq);
    i_seq = linspace(0,1,k);
    I_seq = linspace(0,1,80);
    X_seq = spline(i_seq,x_seq,I_seq);
    Y_seq = spline(i_seq,y_seq,I_seq);
    Z_seq = spline(i_seq,z_seq,I_seq);
    path(i).data = [X_seq', Y_seq', Z_seq'];
end
    

figure
% 画山峰曲面
surf(X,Y,Z)      % 画曲面图
shading flat     % 各小曲面之间不要网格
colormapStr=othercolor(32);%100
colormap(gca,colormapStr);
view(-35,23)%
% colormap summer
% 画路径
hold on
% 画起点和终点
scatter3(startPos(1), startPos(2), startPos(3),50,'ko','MarkerFaceColor','c')
hold on
scatter3(goalPos(1), goalPos(2), goalPos(3),50,'ko','MarkerFaceColor','y')
hold on
%% 起点
threat = startPos;
threat_x = threat(1);
threat_y = threat(2);
threat_z = threat(3);
threat_radius = 5;
[xc,yc,zc]=cylinder(threat_radius,4); % create a unit cylinder
% set the center and height
h=0.0;
xc=xc+threat_x;
yc=yc+threat_y;
zc=zc*h+threat_z;
zc(1,:)=0;%添加
c = mesh(xc,yc,zc); % plot the cylinder
set(c,'edgecolor','flat','facecolor','r','FaceAlpha',.9); % set color and transparency
hold on
plot3(xc(2,:),yc(2,:),zc(2,:),'-')
hold on
fill3(xc(2,:),yc(2,:),zc(2,:),'r')
hold on

%% 终点
threat = goalPos;
threat_x = threat(1);
threat_y = threat(2);
threat_z = threat(3);
threat_radius = 5;
[xc,yc,zc]=cylinder(threat_radius); % create a unit cylinder
% set the center and height
h=0.0;
xc=xc+threat_x;
yc=yc+threat_y;
zc=zc*h+threat_z;
zc(1,:)=0;%添加
c = mesh(xc,yc,zc); % plot the cylinder
set(c,'edgecolor','flat','facecolor','r','FaceAlpha',.9); % set color and transparency
hold on     
plot3(xc(2,:),yc(2,:),zc(2,:),'-')
hold on
fill3(xc(2,:),yc(2,:),zc(2,:),'r')
hold on

%% 
text(startPos(1), startPos(2), startPos(3)+10,'起点','Color','k','FontSize',10)
text(goalPos(1), goalPos(2), goalPos(3)+10,'终点','Color','k','FontSize',10)
% % 画路径
% strcolor=linspecer(num);%获取颜色
leg=[];
for i=1:num
    Pk=plot3(path(i).data(:,1), path(i).data(:,2),path(i).data(:,3),strcolor{i},'LineWidth',2.5);
% Pk=plot3(path(i).data(:,1), path(i).data(:,2),path(i).data(:,3),'color',strcolor(i,:),'LineWidth',2.5);
leg=[leg,Pk];
end
xlabel('x')
ylabel('y')
zlabel('z')
legend(leg,LengendStr,'location','best')
hold off
grid on
end

