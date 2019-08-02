
clear;
close all

%% Data loading
load = 'big';                %% set to 'small' or 'big'
Data = LoadData(load);
[Data.train.X, Data.val.X, Data.test.X] = PreProcess(Data);

%% Parameter settings
use_bn = 1; batch_size = 100; m = [50, 50]; lambda=0.0043;
% Data.train.X = Data.train.X(1:10, :);  %%% only for gradient testing

NetParams = InitNetParams(use_bn, Data.train.X, Data.train.Y, m);

EtasParams.step_size = 5 * NetParams.n_samples / batch_size;
EtasParams.cycles = 2; EtasParams.bottom = 1e-4; EtasParams.top = 1e-2;
TrainParams = InitTrainParams(Data, batch_size, lambda, EtasParams);

%% Gradient test

% [Grads, Errors] = TestGradient(Data.train.X, Data.train.Y, NetParams);

% Exhaustive search for lambdas
l_min = -5; l_max = -1;
lambdas = zeros(1, 20);
    for i = 1 : 20
        l = l_min + (l_max - l_min) * rand(1, 1);
        lambdas(i) = 10^l;
    end
test_accuracies = zeros(1, size(lambdas, 2));
for i = 1:size(lambdas, 2)
    TrainParams.lambda = lambdas(i);
    test_accuracies(i) = MiniBatchGD(Data, TrainParams, NetParams);
end

%% Gradient Decent
step_size =  5 * NetParams.n_samples / batch_size;

plot(TrainParams.etas)
MiniBatchGD(Data, TrainParams, NetParams)


%% Network functions
function [loss, cost] = ComputeCost(X, Y, NetParams, lambda)
    P = EvaluateClassifier(X, NetParams);
    k = length(NetParams.W);
    loss = 0;
    for i = 1:size(X, 2)
         loss = loss - log(Y(:, i)' * P(:, i));
    end
    loss = loss / size(X, 2);
    reg = 0;
    for i = 1:k
        reg = reg + sum(sum(NetParams.W{i}.^2));
    end
    cost = loss + lambda * reg;
end
function [P, NetParams] = EvaluateClassifier(Xin, NetParams)
    alpha = 0.82;
    W = NetParams.W; b = NetParams.b;
    k = length(W);
    X = cell(k + 1, 1); S = cell(k, 1); S_hat = cell(k - 1, 1);
    mu = cell(k - 1, 1); v = cell(k - 1, 1);
   
    X{1} = Xin;  
    if NetParams.use_bn
        for l = 1 : k - 1
            S{l} = W{l} * X{l} + b{l};
            
            if NetParams.train
                mu{l} = mean(S{l}, 2);
                v{l} = var(S{l}, 1, 2);
        
                if isempty(NetParams.mu_avg{l})
                    NetParams.mu_avg{l} = mu{l};
                    NetParams.v_avg{l} = v{l};
                else
                    NetParams.mu_avg{l} = alpha * NetParams.mu_avg{l} + (1 - alpha) * mu{l}; 
                    NetParams.v_avg{l} = alpha * NetParams.v_avg{l} + (1 - alpha) * v{l};
                end
                mu{l} = NetParams.mu_avg{l};
                v{l} = NetParams.v_avg{l};
            else
                mu{l} = NetParams.mu_avg{l};
                v{l} = NetParams.v_avg{l};
            end
            
            S_hat{l} = BatchNormalize(S{l}, mu{l}, v{l});
            S_tilde = NetParams.gammas{l} .* S_hat{l} + NetParams.betas{l};
            X{l + 1} = max(0, S_tilde);
        end
        S{k} = NetParams.W{k} * X{k} + NetParams.b{k};
        P = softmax(S{k});
        
        NetParams.X = X; NetParams.S = S; NetParams.mu = mu; NetParams.v = v;
        NetParams.S_hat = S_hat;

    else
        for i = 1 : k - 1
            S{i} = W{i} * X{i} + b{i};
            X{i+1} = max(0, S{i});
        end
        NetParams.X = X; NetParams.S = S;
        P = softmax(W{k} *  X{k} + b{k});
    end
    
    
end
function acc = ComputeAccuracy(X, y, NetParams)
    P = EvaluateClassifier(X, NetParams);
    [~, argmax] = max(P);
    acc = double(sum(eq(argmax, y)) / length(y));
end
function [Grads, NetParams] = ComputeGradients(Xin, Y, NetParams, lambda)
    Grads.W = cell(numel(NetParams.W), 1);
    Grads.b = cell(numel(NetParams.b), 1);
    N = size(Xin, 2);
    k = numel(Grads.W);
    Grads.gammas = cell(k-1, 1);
    Grads.betas = cell(k-1, 1);
    
    %% Forward pass
    [P, NetParams] = EvaluateClassifier(Xin, NetParams);
 
    W = NetParams.W; 
%     mu = NetParams.mu_avg; v = NetParams.v_avg;
    X = NetParams.X; 
    
    %% Backprop
    G = (P-Y);

    if NetParams.use_bn
        S = NetParams.S; S_hat = NetParams.S_hat; gammas = NetParams.gammas;
        mu = NetParams.mu_avg; v = NetParams.v_avg;
        Grads.W{k} = G * X{k}' / N + 2 * lambda * W{k};
        Grads.b{k} = G / N * ones(N, 1);
        G = W{k}' * G;
        G = G .* (X{k} > 0);
        for l = k - 1 : -1 : 1
            Grads.gammas{l} = G .* S_hat{l} * ones(N, 1) / N;
            Grads.betas{l} = G * ones(N, 1) / N;
            G = G .*  (gammas{l} * ones(N, 1)');
            G = BatchNormBackPass(G, S{l}, mu{l}, v{l});
            Grads.W{l} = G * X{l}' / N + 2 * lambda * W{l};
            Grads.b{l} = G / N * ones(N, 1);
%             Grads.b{l} = Grads.b{l} .* (abs(Grads.b{l}) > 1e-9);
            if l > 1
                G = W{l}' * G;
                G = G .* (X{l} > 0);
            end
        end
    else
        for i = k : -1 : 2
            Grads.W{i} = G * X{i}' / N + 2 * lambda * NetParams.W{i};
            Grads.b{i} = G / N * ones(N, 1);
            G = NetParams.W{i}' * G;
            G = G .* (X{i} > 0);
        end
        Grads.W{1} = G * X{1}' / N + 2 * lambda * NetParams.W{1};
        Grads.b{1} = G / N * ones(N, 1);
    end
    
end    
function Grads = ComputeGradsNumSlow(X, Y, NetParams, lambda, h)
Grads.W = cell(numel(NetParams.W), 1);
Grads.b = cell(numel(NetParams.b), 1);
if NetParams.use_bn
    Grads.gammas = cell(numel(NetParams.gammas), 1);
    Grads.betas = cell(numel(NetParams.betas), 1);
end

for j=1:length(NetParams.b)
    Grads.b{j} = zeros(size(NetParams.b{j}));
    NetTry = NetParams;
    for i=1:length(NetParams.b{j})
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) - h;
        NetTry.b = b_try;
        [~, c1] = ComputeCost(X, Y, NetTry, lambda);        
        
        b_try = NetParams.b;
        b_try{j}(i) = b_try{j}(i) + h;
        NetTry.b = b_try;        
        [~, c2] = ComputeCost(X, Y, NetTry, lambda);
        
        Grads.b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(NetParams.W)
    Grads.W{j} = zeros(size(NetParams.W{j}));
        NetTry = NetParams;
    for i=1:numel(NetParams.W{j})
        
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) - h;
        NetTry.W = W_try;        
        [~, c1] = ComputeCost(X, Y, NetTry, lambda);
    
        W_try = NetParams.W;
        W_try{j}(i) = W_try{j}(i) + h;
        NetTry.W = W_try;        
        [~, c2] = ComputeCost(X, Y, NetTry, lambda);
    
        Grads.W{j}(i) = (c2-c1) / (2*h);
    end
end

if NetParams.use_bn
    for j=1:length(NetParams.gammas)
        Grads.gammas{j} = zeros(size(NetParams.gammas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.gammas{j})
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) - h;
            NetTry.gammas = gammas_try;        
            [~, c1] = ComputeCost(X, Y, NetTry, lambda);
            
            gammas_try = NetParams.gammas;
            gammas_try{j}(i) = gammas_try{j}(i) + h;
            NetTry.gammas = gammas_try;        
            [~, c2] = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.gammas{j}(i) = (c2-c1) / (2*h);
        end
    end
    
    for j=1:length(NetParams.betas)
        Grads.betas{j} = zeros(size(NetParams.betas{j}));
        NetTry = NetParams;
        for i=1:numel(NetParams.betas{j})
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) - h;
            NetTry.betas = betas_try;        
            [~, c1] = ComputeCost(X, Y, NetTry, lambda);
            
            betas_try = NetParams.betas;
            betas_try{j}(i) = betas_try{j}(i) + h;
            NetTry.betas = betas_try;        
            [~, c2] = ComputeCost(X, Y, NetTry, lambda);
            
            Grads.betas{j}(i) = (c2-c1) / (2*h);
        end
    end    
end
end
function S_hat = BatchNormalize(S, mu, v)
    S_hat = (S - mu) .* ((v + eps).^(-0.5));
%     S_hat = diag((v + eps).^(-0.5))*(S - repmat(mu, 1, size(S, 2)));
end
function G = BatchNormBackPass(G, S, mu, v)

    N = size(S, 2);
    sigma1 = (v + eps).^(-0.5);
    sigma2 = (v + eps).^(-1.5);
    G1 = G .* sigma1;
    G2 = G .* sigma2;
    D =  S - mu;
    c =  (G2 .* D) * ones(N, 1);
    G = G1 - G1 * ones(N, 1) * ones(N, 1)' / N - (D  .* c) / N;

end
function p = softmax(x)
%SOFTMAX Softmax function that avoids overflow.
% 
%   P = SOFTMAX(X), X can be a vector, a matrix, or a MxNxK array. If X is
%   a matrix, SOFTMAX calculates the softmax along the first dimension
%   (matrix columns). If X is a MxNxK array, it calculates the softmax
%   along the third dimension.
% 
% See also: exp, bsxfun
% 
% Stavros Tsogkas, <stavros.tsogkas@centralesupelec.fr>
% Last update: March 2015 

if isvector(x) 
    expx = exp(x-max(x));
    p = expx ./ sum(expx);
elseif ismatrix(x)
    expx = exp(bsxfun(@minus, x, max(x,[],1)));
    p = bsxfun(@rdivide, expx, sum(expx,1));
elseif size(x,3) > 1
    expx = exp(bsxfun(@minus, x, max(x,[],3)));
    p = bsxfun(@rdivide, expx, sum(expx,3));
else
    error('Input must be either a vector, a matrix, or a MxNxK array')
end
end

%% Training
function test_acc = MiniBatchGD(Data, TrainParams, NetParams)
    batch_size = TrainParams.batch_size;
    X = Data.train.X; Y = Data.train.Y; y = Data.train.y;
    Xval = Data.val.X; Yval = Data.val.Y; yval = Data.val.y;
    Xtest = Data.test.X; Ytest = Data.test.Y; ytest = Data.test.y;
    [d, n_samples] = size(X);
    K = size(Y, 1);
    n_epochs = TrainParams.epochs; etas = TrainParams.etas; lambda = TrainParams.lambda;
 
    
    costs = zeros(1, n_epochs); val_costs = zeros(1, n_epochs);
    losses = zeros(1, n_epochs); val_losses = zeros(1, n_epochs);
    accuracies = zeros(1, n_epochs); val_accuracies = zeros(1, n_epochs);
    t = 1;
    k = numel(NetParams.W);
    
    
    for epoch = 1:n_epochs
        X(:, randperm(n_samples)); 
        NetParams.train = true;
        for j = 1:n_samples / batch_size
            eta = etas(t);
            t = t + 1;
            inds = (j-1)*batch_size + 1 : j*batch_size;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            
            [grads, NetParams] = ComputeGradients(Xbatch, Ybatch, NetParams, lambda);
            
            for layer = 1:k
                NetParams.W{layer} = NetParams.W{layer} - eta*grads.W{layer};
                NetParams.b{layer} = NetParams.b{layer} - eta*grads.b{layer};
                if NetParams.use_bn && layer < k - 1
                    NetParams.betas{layer} = NetParams.betas{layer} - eta*grads.betas{layer};
                    NetParams.gammas{layer} = NetParams.gammas{layer} - eta*grads.gammas{layer};
                
                end
            end       
        end
        NetParams.train = false;
        [losses(epoch), costs(epoch)] = ComputeCost(X, Y, NetParams, lambda);
        [val_losses(epoch), val_costs(epoch)] = ComputeCost(Xval, Yval, NetParams, lambda);
        accuracies(epoch) = ComputeAccuracy(X, y, NetParams);
        val_accuracies(epoch) = ComputeAccuracy(Xval, yval, NetParams);
        printlist = [losses(epoch), val_losses(epoch), costs(epoch), val_costs(epoch), accuracies(epoch), val_accuracies(epoch)];
        fprintf('Loss: %2.2f; Validation Loss: %2.2f; Cost: %4.2f; Validation Cost: %4.2f; Accuracy: %4.2f; Validation Accuracy %4.2f;\n', printlist)
    end
    
    
    subplot(1, 3, 1)
    hold on
    title('Costs')
    axis tight
    xlabel('Epoch')
    ylabel('cost')
    plot(costs)
    plot(val_costs)
    legend('training', 'validation');
    
    subplot(1, 3, 2)
    hold on
    title('Losses')
    axis tight
    xlabel('Epoch')
    ylabel('loss')
    plot(losses)
    plot(val_losses)
    legend('training', 'validation');
    
    subplot(1, 3, 3)
    hold on
    title('Accuracies')
    axis tight
    xlabel('Epoch')
    ylabel('Accuracy')
    plot(accuracies)
    plot(val_accuracies)
    legend('training', 'validation');
    
    test_acc = ComputeAccuracy(Xtest, ytest, NetParams)
%     
%     print('~/Documents/MATLAB/Deep Learning/ResultPics/6cost_accuracy.pdf', '-dpdf', '-bestfit')
end


%% Data loading and preprocessing
function Data = LoadData(load)

    if strcmp(load, 'big')
        [train.X, train.Y, train.y] = LoadBatch(1);
        for i = 2:4
            [X ,Y, y] = LoadBatch(i);
            train.X = [train.X, X];
            train.Y = [train.Y, Y];
            train.y = [train.y, y];
        end
        [X5, Y5, y5] = LoadBatch(5);
        train.X = [train.X, X5(:, 1:5000)];
        train.Y = [train.Y, Y5(:, 1:5000)];
        train.y = [train.y, y5(:, 1:5000)];
    
        val.X = X5(:, 5001:10000);
        val.Y = Y5(:, 5001:10000);
        val.y = y5(:, 5001:10000);
    end
    
    if strcmp(load, 'small')
        [train.X, train.Y, train.y] = LoadBatch(1);
        [val.X, val.Y, val.y] = LoadBatch(2);
    end
    [test.X, test.Y, test.y] = LoadBatch(0);
    Data.train = train; Data.val = val; Data.test = test;
end
function [X, Y, y] = LoadBatch(batch_number)
    addpath datasets/cifar-10-batches-mat/;
    if batch_number == 0
        filename = 'test_batch.mat';
    else
    
        filename = strcat('data_batch_', string(batch_number), '.mat');
    end

    file = load(filename);
    
    X = double(file.data)' / 255;
    y = (file.labels + 1)';
    Y = double(zeros(10, length(y)));
    for i = 1:length(y)
        Y(y(i),i) = 1;
    end

    
end
function  [Xtrain, Xval, Xtest] = PreProcess(Data)
    
    norm = mean(Data.train.X, 2);
    Xtrain = Data.train.X - norm;
    Xval = Data.val.X - norm;
    Xtest = Data.test.X - norm;
end

%% Plotting

%% Parameters
function NetParams = InitNetParams(use_bn, X, Y, m)
    [d, NetParams.n_samples] = size(X);
    K = size(Y, 1);
    NetParams.d = d;
    NetParams.K = K;
    NetParams.use_bn = use_bn;
    NetParams.k = numel(m) + 1;
    k = NetParams.k;
    
    W = cell(k, 1);
    b = cell(k, 1);
    gammas = cell(k-1, 1);
    betas = cell(k-1, 1);
    NetParams.mu_avg = cell(k-1, 1);
    NetParams.v_avg = cell(k-1, 1);
    rng(400);

    %% Input layer
%     W{1} = double(1/sqrt(d)) * double(randn(m(1), d));
    W{1} = HE_init(m(1), d);
    b{1} = double(zeros(m(1), 1));
    gammas{1} = ones(m(1), 1);
    betas{1} = zeros(m(1), 1);
%     gammas{1} = double(randn(m(1), 1));        

    
    for i = 2:k-1
%         W{i} = double(1/sqrt(d)) * double(randn(m(i), m(i-1)));
        W{i} = HE_init(m(i), m(i-1));
        b{i} = double(zeros(m(i), 1));
%         gammas{i} = double(randn(m(i), 1));
        gammas{i} = ones(m(i), 1);
        betas{i} = zeros(m(i), 1);
    end
    
    %% Output layer
%     W{k} = double(1/sqrt(m(end)) * double(randn(K, m(end))));
    W{k} = HE_init(K, m(end));
    b{k} = double(zeros(K, 1));
    
    %% Set Output
    NetParams.W = W;  NetParams.b = b;
    NetParams.gammas = gammas; NetParams.betas = betas;
    
end
function TrainParams = InitTrainParams(Data, batch_size, lambda, EtasParams)
    TrainParams.batch_size = batch_size; TrainParams.lambda = lambda;
    N = size(Data.train.X, 2);
    [TrainParams.epochs, TrainParams.etas] = CyclicEtas(N, batch_size, EtasParams);
end
function [epochs, etas] = CyclicEtas(N, batch_size, EtasParams)
    cycles = EtasParams.cycles;
    epochs = 2 * EtasParams.step_size * cycles * batch_size / N;
    k = (EtasParams.top - EtasParams.bottom) / EtasParams.step_size;
    t = 1:EtasParams.step_size;
    half_cycle = EtasParams.bottom + k * t;
    full_cycle = [half_cycle, fliplr(half_cycle)];
    etas = [];
    for i = 1:cycles
        etas = [etas, full_cycle];
    end
end
function param = HE_init(out, in)
    param = double(randn(out, in)) * sqrt(double(2) / double(in)) ;
end
    



%% Tests
function [abs, rel] = CalcErrors(grad_a, grad_b)
    abs = norm(grad_a - grad_b);
    rel = abs / max(eps, norm(grad_a) + norm(grad_b));
end
function [Grads, Errors] = TestGradient(Xtrain, Ytrain, NetParams)
lambda = 0;
NetParams.train = true;
n_samples = 100;
Grads.a = ComputeGradients(Xtrain(:, 1:n_samples), Ytrain(:, 1:n_samples), NetParams, lambda);
Grads.n = ComputeGradsNumSlow(Xtrain(:, 1:n_samples), Ytrain(:, 1:n_samples), NetParams, lambda, 1e-5);
k = NetParams.k;
Errors.abs.W = cell(k, 1);
Errors.rel.W = cell(k, 1);
Errors.abs.b = cell(k, 1);
Errors.rel.b = cell(k, 1);
Errors.abs.gammas = cell(k-1, 1);
Errors.rel.gammas = cell(k-1, 1);
Errors.abs.betas = cell(k-1, 1);
Errors.rel.betas = cell(k-1, 1);
for i = 1 : k
    [Errors.abs.W{i}, Errors.rel.W{i}] = CalcErrors(Grads.a.W{i, 1}, Grads.n.W{i, 1});
    [Errors.abs.b{i}, Errors.rel.b{i}] = CalcErrors(Grads.a.b{i, 1}, Grads.n.b{i, 1});   
end
if NetParams.use_bn
    for i = 1 : k - 1
        [Errors.abs.gammas{i}, Errors.rel.gammas{i}] = CalcErrors(Grads.a.gammas{i, 1}, Grads.n.gammas{i, 1});
        [Errors.abs.betas{i}, Errors.rel.betas{i}]  = CalcErrors(Grads.a.betas{i, 1}, Grads.n.betas{i, 1});
    end
end

end
