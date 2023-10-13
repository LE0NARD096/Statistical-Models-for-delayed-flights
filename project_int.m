clear; clc; close all;

%% loading
data            = readtable('NewFlightDelaysTraining.csv');
datatest        = readtable('NewFlightDelaysTest.csv');
Xname   =data.Properties.VariableNames;
data    =table2cell(data);
datatest=table2cell(datatest);
X       = cellfun(@double,data);
T       = cellfun(@double,datatest);
%hasNaN = any(isnan(X(:)));
%plotmatrix(X);

%% new columns    
DelayDep   = (X(:,17)-X(:,1)>15);
LowDist = (X(:,18)<=199);
MorningFl= (X(:,1)<=1400);
DelayDepT   = (T(:,17)-T(:,1)>15);
LowDistT = (T(:,18)<=199);
MorningFlT= (T(:,1)<=1400);

%% preprocessing
Xtr     = [X(:,2:16) DelayDep LowDist MorningFl X(:,20:21)];
XtrName = [Xname(:,2:16) 'DelayDep' 'LowDist' 'MorningFl' Xname(:,20:21)];
Xtot    = [Xtr(:,1:19)];
ytot    = [Xtr(:,20)];

Ttot    = [T(:,2:16) DelayDepT LowDistT MorningFlT T(:,20)];

%% dataset splitting
[N,p]       = size(Xtr);
R           = cvpartition(N,'Holdout',0.30);
obsTest     = test(R);
y           = Xtr(:,20);
Xtr_tr      = Xtr(~obsTest,1:19); 
Xtr_trname  = [XtrName(1:19)];
Xtr_te      = Xtr(obsTest,1:19); 
y_tr        = y(~obsTest,:); 
y_te        = y(obsTest,:); 

for i = 1:19
        importance(i) = sum(Xtr_tr(:,i) == y_tr(:,1)) / numel(y_tr);
end

%% logistic regression
mdlLogistic             = fitglm(Xtr_tr,y_tr, 'Distribution', 'binomial', 'Link', 'logit');
yhatLR                  = predict(mdlLogistic,Xtr_te);
yhatLRdelay             = predict(mdlLogistic,Xtr_te)> 0.5;
[xVallr,yVallr,~,auclr] = perfcurve(y_te,yhatLR,'1');
plot(xVallr,yVallr);
yhatLRdelaya = double(yhatLRdelay);
confusionchart(y_te,yhatLRdelaya);
%confusionmat(y_te,yhatLRdelaya);

%% logistic regression total (cheat)
mdlLogisticT              = fitglm(Xtot,ytot, 'Distribution', 'binomial', 'Link', 'logit');
yhatLRtot                 = predict(mdlLogisticT,Ttot);
yhatLRdelaytot            = predict(mdlLogisticT,Ttot)> 0.5;

%% naive Bayes
mdlNB                   = fitcnb(Xtr_tr(:,:),y_tr,'DistributionNames',{'mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn','mvmn'});
[yhatNB,post]           = predict(mdlNB,Xtr_te(:,:));
[CMnaive,order]         = confusionmat(y_te,yhatNB);
[xValnb,yValnb,~,aucNB] = perfcurve(y_te,yhatNB,'1');

%% discriminant analysis 
mdlQDA                     = fitcdiscr(Xtr_tr,y_tr,'DiscrimType','linear');
[yhatQDA,postQDA]          = predict(mdlQDA,Xtr_te);
[xValqda,yValqda,~,aucqda] = perfcurve(y_te,yhatQDA,'1');


%% Lasso classification
[B,FitInfo] = lassoglm(Xtr_tr,y_tr,'binomial','CV',5);
lassoPlot(B,FitInfo,'plottype','CV');
idxLambdaMinDeviance = FitInfo.IndexMinDeviance;
B0 = FitInfo.Intercept(idxLambdaMinDeviance);
coef = [B0; B(:,idxLambdaMinDeviance)];
yhatLasso = glmval(coef,Xtr_te,'logit');
[xVallas,yVallas,~,aucLas] = perfcurve(y_te,yhatLasso,'1');
plot(xVallas,yVallas);

%% classification tree
AutoTree = fitctree(Xtr_tr,y_tr,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
yhatAutotree            = predict(AutoTree,Xtr_te);
[xValat,yValat,~,aucat] = perfcurve(y_te,yhatAutotree,'1');
tree = fitctree(Xtr_tr,y_tr);

%% bagging
%leaf = [5 10 20 50 100];
%col = 'rbcmy';
%figure
%hold on
%for i=1:length(leaf)
%    b = TreeBagger(500,Xtr_tr,y_tr,'Method','classification', ...
%        'OOBPrediction','On', ...
%        'MinLeafSize',leaf(i));
%    plot(oobError(b),col(i))
%end
%xlabel('Number of Grown Trees')
%ylabel('Classification Error') 
%legend({'5' '10' '20' '50' '100'},'Location','NorthEast')
%hold off

b = TreeBagger(500,Xtr_tr,y_tr,'Method','classification', ...
    'OOBPredictorImportance','On', ...
    'MinLeafSize',5);
figure
bar(b.OOBPermutedPredictorDeltaError)
xlabel('Feature Number') 
ylabel('Out-of-Bag Feature Importance')
yhatbag         = predict(b,Xtr_te);
yhatbag         = str2num(cell2mat(yhatbag));
[xValbag,yValbag,~,aucbag] = perfcurve(y_te,yhatbag,'1');



