clear all;	
warning('off', 'all');

addpath(genpath('../liblinear/matlab'));
%addpath(genpath('../libsvm/matlab'));
addpath(genpath('../ml'));
addpath(genpath('./tSNE_matlab'));


pcaK = 100;              % subspace dims
k = 128;                % number of basis vectors
mu = 10000;           % MMD regularization
alpha = 1;            % graph regularization
lambda = 0.01;           % sparsity regularization
nIters = 10;            % number of iterations

runs = 1;               % number of repeated runs
result = [];
%para={10,20,50,80,100,120,200,500,800,1000};

for i=1:runs%length(para)
    %k=para{i};
    %load the demo data or call genData to randomly load
    load('amazon-decaf(PCA)-to-webcam-surf.mat');

    % [S,S_Label,~,~]=genData('amazon','decaf','S',20);
    % S=double(S);
    % [Ttest,Ttest_Label,T,T_Label]=genData('webcam','surf','T',3);%Caltech10 webcam
    % T=double(T);Ttest=double(Ttest);

    % %  Align the dimensions by PCA  

     % [Ps,~,~] = princomp(S);
     % S = S*Ps(:,1:800);
%     
%     [Ptest,~,~] = princomp(Ttest);
%     Ttest = Ttest*Ptest(:,1:800);
%     
%     [Pt,~,~] = princomp(T);
%     T = T*Pt(:,1:800);
    
   % clear Ps Ptest Pt;
    
    Xs = sparse(S);
    Xt = sparse(Ttest);
    Ys = sparse(S_Label);
    Yt = sparse(Ttest_Label);
    Xref=sparse(T);
    Yref=sparse(T_Label);
    
    % Normalization of original data
    Xs = diag(sparse(1./sqrt(sum(Xs.^2,2))))*Xs;
    Xt = diag(sparse(1./sqrt(sum(Xt.^2,2))))*Xt;
    Xref=diag(sparse(1./sqrt(sum(Xref.^2,2))))*Xref;
    
  
    % Perform PCA on original data
    pca_options.ReducedDim = pcaK;
    EigenVec = PCA([Xs;Xref;Xt],pca_options);
    newX = [Xs;Xref;Xt]*EigenVec;
    newXs = newX(1:size(Xs,1),:);
    newXref=newX(size(Xs,1)+1:size(Xs,1)+size(Xref,1),:);
    newXt = newX(size(Xs,1)+size(Xref,1)+1:end,:);
    
    
    newXs = diag(sparse(1./sqrt(sum(newXs.^2,2))))*newXs;
    newXt = diag(sparse(1./sqrt(sum(newXt.^2,2))))*newXt;
    newXref=diag(sparse(1./sqrt(sum(newXref.^2,2))))*newXref;
    newXt=[newXref;newXt];
    
 
    for it = 1:runs
        
        fprintf('start training, random runs %d\n',it);
         
        X = [newXs',newXt'];
        
        B = rand(size(X,1),k)-0.5;
	    B = B - repmat(mean(B,1), size(B,1),1);
        B = B*diag(1./sqrt(sum(B.*B)));
        
        W = constructW(X');
        D = spdiags(full(sum(W,2)),0,speye(size(W,1)));
        L = D - W;
        
        ns = size(newXs',2);
        nt = size(newXt',2);
        e = [1/ns*ones(ns,1);-1/nt*ones(nt,1)];
        M = e*e';
        
        S= learn_coding(B, X, alpha, mu,lambda, L, M);
        for t=2:nIters
           S= learn_coding(B, X, alpha, mu,lambda, L, M, S);
           S(isnan(S))=0;
           Ss = S(:,1:ns);
           St = S(:,ns+1:ns+nt);
           B = learn_dictionary(X, S, 1);
        end
   
        model = train(Yref,sparse(St(:,1:size(Xref,1))'),sprintf('-s 0 -c %f -q 1',100));
        [~,acc,~] = predict(Yt,sparse(St(:,size(Xref,1)+1:end)'),model);
        fprintf('accuracy=%0.4f \n\n',acc);
        
    end
    
end
