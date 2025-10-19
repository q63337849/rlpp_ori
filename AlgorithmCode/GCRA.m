% Greater Cane Rat Algorithm (GCRA)                                       %
%                                                                         %
%  Developed in MATLAB R2020b                                             %
%                                                                         %
%  Designed and Developed: Dr. Ovre Agushaka                              %
%                                                                         %
%         E-Mail: jo.agushaka@science.edu.ng                              %
%                 jefshak@gmail.com                                       %
%                                                                         %
%                                  
%                                                                         %
%  Published paper: Agushaka et al.                                       %
%          A novel algorithm for global optimization: Greater cane rat    %
%          algorithm                                                      %
%_________________________________________________________________________%

function[Alpha_score,Alpha_pos,Convergence]=GCRA(Search_Agents,Max_iterations,Lower_bound,Upper_bound,dimension,objective)
Position=zeros(1,dimension);
Score=inf; 
Alpha_pos=zeros(1,dimension);
Alpha_score=inf; 
Alpha_pos1 = zeros(1, dimension); % 初始化Alpha_pos1
Alpha_score1 = inf; % 初始化Alpha_score1为无穷大

Gcanerats=init(Search_Agents,dimension,Upper_bound,Lower_bound);
Convergence=zeros(1,Max_iterations);
l=1;
for i=1:size(Gcanerats,1)  
        
        Flag4Upper_bound=Gcanerats(i,:)>Upper_bound;
        Flag4Lower_bound=Gcanerats(i,:)<Lower_bound;
        Gcanerats(i,:)=(Gcanerats(i,:).*(~(Flag4Upper_bound+Flag4Lower_bound)))+Upper_bound.*Flag4Upper_bound+Lower_bound.*Flag4Lower_bound;               
        
        fitness=objective(Gcanerats(i,:));
        
        if fitness<Score 
            Score=fitness; 
            Position=Gcanerats(i,:);
            Alpha_pos1=Position;
            Alpha_score=Score;
        end
        
 end

while l<Max_iterations+1
    Alpha_pos=max(Alpha_pos1);
 %    Alpha_pos=Alpha_pos1(nn);
   GR_m=randperm(Search_Agents-1,1); 
   GR_rho=0.5;
   GR_r= Alpha_score-l*(Alpha_score/Max_iterations);
    x = 1;
    y = 4;
    GR_mu = floor((y-x).*rand(1,1) + x);
    GR_c=rand;
    GR_alpha=2*GR_r*rand-GR_r;
    GR_beta=2*GR_r*GR_mu-GR_r;
    
    for i=1:size(Gcanerats,1)
        for j=1:size(Gcanerats,2)  
            Gcanerats(i,j)= (Gcanerats(i,j)+Alpha_pos)/2; 
        end
    end
    for i=1:size(Gcanerats,1)
        for j=1:size(Gcanerats,2)  
           if rand<GR_rho
%                  dd=Alpha_pos;
                 Gcanerats(i,j)= Gcanerats(i,j)+GR_c*(Alpha_pos-GR_r*Gcanerats(i,j)); 
                 Flag4Upper_bound=Gcanerats(i,j)>Upper_bound(j);
                 Flag4Lower_bound=Gcanerats(i,j)<Lower_bound(j);
                 Gcanerats(i,j)=(Gcanerats(i,j).*(~(Flag4Upper_bound+Flag4Lower_bound)))+Upper_bound.*Flag4Upper_bound+Lower_bound.*Flag4Lower_bound;               
        
                 fitness=objective(Gcanerats(i,j));
                  if fitness<Score 
                        Score=fitness; 
                        Position=Gcanerats(i,j);
                  else
                      Gcanerats(i,j)= Gcanerats(i,j)+GR_c*(Gcanerats(i,j)-GR_alpha*Alpha_pos); 
                      Flag4Upper_bound=Gcanerats(i,j)>Upper_bound;
                      Flag4Lower_bound=Gcanerats(i,j)<Lower_bound;
                      Gcanerats(i,j)=(Gcanerats(i,j).*(~(Flag4Upper_bound+Flag4Lower_bound)))+Upper_bound.*Flag4Upper_bound+Lower_bound.*Flag4Lower_bound;               

                       fitness=objective(Gcanerats(i,j));
                      if fitness<Score 
                            Score=fitness; 
                            Position=Gcanerats(i,j);
                      end
                  end
            else
                
                 Gcanerats(i,j)= Gcanerats(i,j)+GR_c*(Alpha_pos-GR_mu*Gcanerats(GR_m,j));
                 Flag4Upper_bound=Gcanerats(i,j)>Upper_bound(j);
                 Flag4Lower_bound=Gcanerats(i,j)<Lower_bound(j);
                 Gcanerats(i,j)=(Gcanerats(i,j).*(~(Flag4Upper_bound+Flag4Lower_bound)))+Upper_bound.*Flag4Upper_bound+Lower_bound.*Flag4Lower_bound;               
        
                 fitness=objective(Gcanerats(i,j));
                  if fitness<Score 
                        Score=fitness; 
                        Position=Gcanerats(i,j);
                  else
                      Gcanerats(i,j)= Gcanerats(i,j)+GR_c*(Gcanerats(GR_m,j)-GR_beta*Alpha_pos); 
                      Flag4Upper_bound=Gcanerats(i,j)>Upper_bound;
                      Flag4Lower_bound=Gcanerats(i,j)<Lower_bound;
                      Gcanerats(i,j)=(Gcanerats(i,j).*(~(Flag4Upper_bound+Flag4Lower_bound)))+Upper_bound.*Flag4Upper_bound+Lower_bound.*Flag4Lower_bound;               

                      fitness=objective(Gcanerats(i,j));
                      if fitness<Score 
                            Score=fitness; 
                            Position=Gcanerats(i,j);
                      end
                  end
            end
            Alpha_pos1=Position;
            Alpha_score=Score;    
                        
        end
    end
    l=l+1;    
    Convergence(l)=Score;
end

function Pos=init(SearchAgents,dimension,upperbound,lowerbound)

Boundary= size(upperbound,2); 
if Boundary==1
    Pos=rand(SearchAgents,dimension).*(upperbound-lowerbound)+lowerbound;
end

if Boundary>1
    for i=1:dimension
        ub_i=upperbound(i);
        lb_i=lowerbound(i);
        Pos(:,i)=rand(SearchAgents,1).*(ub_i-lb_i)+lb_i;
    end
end
