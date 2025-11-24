load uthararesults1
MM1=reshape(mean(meanmemory_p(:,end-9:end),2),mcases,rhocases)
MN1=reshape(mean(meanneutral_p(:,end-9:end),2),mcases,rhocases)
MW1=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(1); clf; subplot(2,2,1); surf(rho,m,MM1); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(pink); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=1, evolved epi strength');
figure(2); clf; subplot(2,2,1); surf(rho,m,MW1); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=1, mean fitness');

load uthararesults2
MM2=reshape(mean(meanmemory_p(:,end-9:end),2),mcases,rhocases)
MN2=reshape(mean(meanneutral_p(:,end-9:end),2),mcases,rhocases)
MW2=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(1); subplot(2,2,2); surf(rho,m,MM2); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(pink); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=2, evolved epi strength');
figure(2); subplot(2,2,2); surf(rho,m,MW2); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=2, mean fitness');

load uthararesults3
MM3=reshape(mean(meanmemory_p(:,end-9:end),2),mcases,rhocases)
MN3=reshape(mean(meanneutral_p(:,end-9:end),2),mcases,rhocases)
MW3=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(1); subplot(2,2,3); surf(rho,m,MM3); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(pink); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=3, evolved epi strength');
figure(2); subplot(2,2,3); surf(rho,m,MW3); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=3, mean fitness');

load uthararesults4
MM4=reshape(sum(meanmemory_p(:,end-9:end),2)/10,mcases,rhocases)
MN4=reshape(mean(meanneutral_p(:,end-9:end),2),mcases,rhocases)
MW4=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(1); subplot(2,2,4); surf(rho,m,MM4); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(pink); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=4, evolved epi strength');
figure(2); subplot(2,2,4); surf(rho,m,MW4); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=4, mean fitness');

load uthararesults01
MW01=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(3); clf; subplot(2,2,1); surf(rho,m,MW01); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=1, old-fashioned mean fitness');

load uthararesults02
MW02=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(3); subplot(2,2,2); surf(rho,m,MW02); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=2, old-fashioned mean fitness');

load uthararesults03
MW03=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(3); subplot(2,2,3); surf(rho,m,MW03); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=3, old-fashioned mean fitness');

load uthararesults04
MW04=reshape(mean(meanw(:,end-9:end),2),mcases,rhocases)
figure(3); subplot(2,2,4); surf(rho,m,MW04); view([0 90]); axis([0 1 0.5 0.95]); caxis([0 1]); colormap(copper); colorbar; xlabel('rho, i.e. autocorr'); ylabel('m'); title('dim=4, old-fashioned mean fitness');

% since the above shows that old fashioned stuff produces roughly as good
% fitness as epi-fitness, let's prove that

figure(4); tiledlayout(2,2)
nexttile; plot(MW01,MW1,'bo',[0 1],[0 1]); xlabel('old fashioned fitness'); ylabel('fitness with memory');
nexttile; plot(MW02,MW2,'bo',[0 1],[0 1])
nexttile; plot(MW03,MW3,'bo',[0 1],[0 1])
nexttile; plot(MW04,MW4,'bo',[0 1],[0 1])

% and let's compare to the neutral traits. Turns out there's not so much
% evidence that memory evolves to be higher than neutral. Often it evolves
% to be lower so they're really trying to rely on genes only!
figure(5); tiledlayout(2,2)
nexttile; plot(MN1,MM1,'bo',[0 1],[0 1]); xlabel('neutral memorylike trait'); ylabel('evolved memory');
nexttile; plot(MN2,MM2,'bo',[0 1],[0 1])
nexttile; plot(MN3,MM3,'bo',[0 1],[0 1])
nexttile; plot(MN4,MM4,'bo',[0 1],[0 1])

% a more detailed look at everything
load uthararesults1
for i=1:nof_scenarios
    i
    figure(6); tiledlayout(2,1);
    nexttile; plot(meanw(i,:)); ylabel('Mean fitness'); axis([1 maxgen 0 1]);
    nexttile; plot(1:maxgen-1,meanmemory_g(i,:),'r',1:maxgen-1,meanneutral_g(i,:),'b'); ylabel('Mean memory'); xlabel('generation');
    title(sprintf(['rho = ' num2str(rho(i)) ', m = ' num2str(m(i)) ', alpha = ' num2str(alpha(i)) ', beta = ' num2str(beta(i))]));
    pause
end
