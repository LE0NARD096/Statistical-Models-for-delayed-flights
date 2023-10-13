clear; clc; close all;

for m=1:3
    for n=1:3
        parfor i = 1:200
        p=8;
        N=[100 500 1000];
        b           = [1 1 1 1 1 1 1 1];
        bb          = [1 1 1 1 0 0 0 0];
        ro          = [0 0.5 0.95];
        cov_matrix  = eye(p) + ro(m)*(ones(p)-eye(p));
        X           = mvnrnd(zeros(N(n),p),cov_matrix);
        %cov(X);
        eps         = normrnd(0,sqrt(3),N(n),1);
        Y           = X*b'  + eps;
        YY          = X*bb' + eps;
        [N,p]       = size(X);
        R           = cvpartition(N,'Holdout',0.30);
        obsTest     = test(R);
        X_tr        = X(~obsTest,:); 
        X_te        = X(obsTest,:); 
        y_tr        = Y(~obsTest,:); 
        y_te        = Y(obsTest,:); 
        yy_tr       = YY(~obsTest,:); 
        yy_te       = YY(obsTest,:); 

        %% Linear
        %mod1        = fitlm(X_tr,y_tr,"linear");
        %mod2        = fitlm(X_tr,yy_tr,"linear");
        %Yhat        = predict(mod1,X_te);
        %YYhat       = predict(mod2,X_te);

        %% Stepwise Model Selection
        %fun           = @(XTRAIN, YTRAIN, XTEST,YTEST) norm(YTEST-XTEST*regress(YTRAIN,XTRAIN))^2;
        %[inB,historyB]  = sequentialfs(fun,X_tr,y_tr,'cv',10, 'direction','forward');
        %ForwardMdlB    = fitlm(X_tr(:,inB),y_tr);
        %ypredForwardB  = predict(ForwardMdlB,X_te(:,inB));
        %[inBB,historyBB]  = sequentialfs(fun,X_tr,yy_tr,'cv',10, 'direction','forward');
        %ForwardMdlBB    = fitlm(X_tr(:,inBB),yy_tr);
        %ypredForwardBB  = predict(ForwardMdlBB,X_te(:,inBB));
        %er(i)         = mean((y_te - ypredForwardB).^2);
        %err(i)        = mean((yy_te - ypredForwardBB).^2);

        %% LASSO
        [BlassoB, Stats] = lasso(X_tr,y_tr,'CV', 5);
        BcvB         = BlassoB(:,Stats.IndexMinMSE);
        [BlassoBB, Stats] = lasso(X_tr,yy_tr,'CV', 5);
        BcvBB         = BlassoBB(:,Stats.IndexMinMSE);
        indsparB     = find(BcvB~=0);
        indsparBB     = find(BcvBB~=0);
        LassoMdlB    = fitlm(X_tr(:,indsparB),y_tr);
        ypredLassoB  = predict(LassoMdlB,X_te(:,indsparB));
        LassoMdlBB    = fitlm(X_tr(:,indsparBB),yy_tr);
        ypredLassoBB  = predict(LassoMdlBB,X_te(:,indsparBB));
        er(i)       = mean((y_te - ypredLassoB).^2);
        err(i)      = mean((yy_te - ypredLassoBB).^2);

        %% Ridge
        %BRB         = ridge(y_tr,X_tr,500,0);
        %ypredRidgeB = [ones(size(y_te)) X_te]*BRB;
        %BRBB         = ridge(yy_tr,X_tr,500,0);
        %ypredRidgeBB = [ones(size(y_te)) X_te]*BRBB;
        %er(i)       = mean((y_te - ypredRidgeB).^2);
        %err(i)      = mean((yy_te - ypredRidgeBB).^2);

        %% PCA
        %[PCALoadings,PCAScores,PCAVar,tsquared,explained] = pca(X_tr,'Economy',false);
        %VarExpl = cumsum(PCAVar)./sum(PCAVar);
        %Varaeach = PCAVar./sum(PCAVar);
        %betaPCRB                            = regress(y_tr, PCAScores(:,1:8));
        %betaPCRB                            = PCALoadings(:,1:8)*betaPCRB;
        %betaPCRB                            = [mean(y_tr); betaPCRB];
        %betaPCRBB                           = regress(yy_tr, PCAScores(:,1:8));
        %betaPCRBB                           = PCALoadings(:,1:8)*betaPCRBB;
        %betaPCRBB                           = [mean(yy_tr); betaPCRBB];
        %ypredPCRB                           = [ones(size(y_te)) X_te]*betaPCRB;
        %ypredPCRBB                          = [ones(size(yy_te)) X_te]*betaPCRBB;
        %er(i)       = mean((y_te - ypredPCRB).^2);
        %err(i)      = mean((yy_te - ypredPCRBB).^2);
        %pareto(Varaeach)
        end
    ER(n,:)=er;
    ERR(n,:)=err;
    end
ERROREmodDENSO{m}=ER;
ERROREmodSPARSO{m}=ERR;
end


figure;
subplot(2,3,1,'replace')
boxplot(ERROREmodDENSO{1, 1}')
title(['$\rho = 0.$'])
xlabel('Observations')
ylabel('MSE')

subplot(2,3,2)
boxplot(ERROREmodDENSO{1, 2}')
title(['$\rho = 0.5$'])
xlabel('Observations') 
ylabel('MSE')

subplot(2,3,3)
boxplot(ERROREmodDENSO{1, 3}')
title(['$\rho = 0.95$'])
xlabel('Observations') 
ylabel('MSE')

subplot(2,3,4)
boxplot(ERROREmodSPARSO{1, 1}')
title(['$\rho = 0.$'])
xlabel('Observations')
ylabel('MSE')

subplot(2,3,5)
boxplot(ERROREmodSPARSO{1, 2}')
title(['$\rho = 0.5$'])
xlabel('Observations') 
ylabel('MSE')

subplot(2,3,6)
boxplot(ERROREmodSPARSO{1, 3}')
title(['$\rho = 0.95$'])
xlabel('Observations') 
ylabel('MSE')

