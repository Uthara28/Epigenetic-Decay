clear;
tic;

load world; % World caters for max 1000 generations. Shorter than that is possible below, but not longer

popsize=5000;
maxgen=1000;
mutrate=[0.01 0.01]; % first mutation is for epi-memory (and the neutral "memory"), 2nd for geno
mutsize=[1 1]; % ditto
epsilon=0.1;
nof_dimensions=4; % anything from 1 upwards

% each scenario is a row, each individual is a column, and each layer is a
% different property of that individual

memory=1;
neutral=2;
geno=3:2+nof_dimensions;
pheno=3+nof_dimensions:2+2*nof_dimensions;
dev=3+2*nof_dimensions:2+3*nof_dimensions;

% initialize all populations
N=nan(nof_scenarios,popsize,dev(end));
N(:,:,memory:neutral)=-3;
N(:,:,geno)=zeros(nof_scenarios,popsize,nof_dimensions);
N(:,:,pheno)=N(:,:,geno)+epsilon*randn(nof_scenarios,popsize); % initial pop has no memories yet
N(:,:,dev)=nan;

% now simulate the hell out of everything
for t=1:maxgen-1
    if t/10==floor(t/10) t, end
    % compute the fitness values based on FGM. In environment A, all optima
    % are zero, in B all optima are 1, thus the value of E gives the
    % optimum whether we are in A or B, which is handy
    for d=1:nof_dimensions
       N(:,:,dev(d))=N(:,:,pheno(d))-E(:,t)*ones([1 popsize]);
       w=exp(-sum(N(:,:,dev).^2,3)); % the 3 here indicates we want to sum across layers (the different dimensions), not individuals or environments
    end

    % collect data
    meanw(:,t)=mean(w,2);
    meanmemory_g(:,t)=mean(N(:,:,memory),2);
    meanmemory_p(:,t)=mean(1./(1+exp(-N(:,:,memory))),2);
    meanneutral_g(:,t)=mean(N(:,:,neutral),2);
    meanneutral_p(:,t)=mean(1./(1+exp(-N(:,:,neutral))),2);

    % now create the next generation: offspring are formed according to
    % fitness of the parental generation;
    for scenario=1:nof_scenarios % sadly this particular bit cannot be vectorized easily (without writing a new randsample function, that is)
        parents=randsample(popsize,popsize,true,w(scenario,:));
        offspring(scenario,:,:)=N(scenario,parents,:);
    end
    % mutate offspring
    mutate_memories=rand([nof_scenarios popsize])<mutrate(1);
    mutate_neutral=rand([nof_scenarios popsize])<mutrate(1);
    mutate_geno=rand([nof_scenarios popsize nof_dimensions])<mutrate(2);
    offspring(:,:,memory)=offspring(:,:,memory)+mutate_memories.*(mutsize(1)*randn(nof_scenarios,popsize));
    offspring(:,:,neutral)=offspring(:,:,neutral)+mutate_neutral.*(mutsize(1)*randn(nof_scenarios,popsize));
    offspring(:,:,geno)=offspring(:,:,geno)+mutate_geno.*(mutsize(2)*randn(nof_scenarios,popsize,nof_dimensions));
    % form their phenotypes (we still know the parents)
    for i=1:nof_dimensions
        offspring(:,:,pheno(i))=offspring(:,:,geno(i))+epsilon*randn(nof_scenarios,popsize);%+1./(1+exp(-offspring(:,:,memory))).*(N(:,parents,pheno(i))-N(:,parents,geno(i)))
    end
    N=offspring;

end
toc

save uthararesults04






