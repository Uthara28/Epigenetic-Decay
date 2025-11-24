clear;
rhocases=20; mcases=21; maxgen=1000;
rho=linspace(0,0.99,rhocases); % autocorr
m=linspace(0.5,0.95,mcases); % mean time system spends in state A

[rho,m]=meshgrid(rho,m);
beta=(1-rho).*m; % probability of environment B changing to A
alpha=1-rho-beta; % probability of environment A changing to B

if all([min(alpha(:)) max(alpha(:)) min(beta(:)) max(beta(:))]>0) && all([min(alpha(:)) max(alpha(:)) min(beta(:)) max(beta(:))]<1)
    disp('all environments are possible')
else
    disp('think about what to do with non-probability alpha or beta values')
    pause
end

% now make every scenario a row, this'll allow us to vectorize better
scenario=[alpha(:) beta(:)]; nof_scenarios=size(scenario,1);

% create the environmental sequences
E(:,1)=zeros([nof_scenarios 1]); % this is environment A
for t=1:maxgen-1
    E(:,t+1)=E(:,t); % baseline: no change
    cAB=find(E(:,t)==0 & rand(nof_scenarios,1)<alpha(:)); % these will want to change from A to B
    cBA=find(E(:,t)==1 & rand(nof_scenarios,1)<beta(:)); % these will want to change from B to A
    E(cAB,t+1)=1;
    E(cBA,t+1)=0;
end

save world
