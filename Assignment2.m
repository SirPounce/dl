 clear;
close all

% [Xtrain, ...
%     Ytrain, ...
%     ytrain] = LoadBatch(1);
% [Xval, ...
%     Yval, ...
%     yval] = LoadBatch(2);
% [Xtest, ...
%     Ytest, ...
%     ytest] = LoadBatch(0);



[Xtrain, Ytrain, ytrain, ...
Xval, Yval, yval, ...
Xtest, Ytest, ytest] = BigLoad();

[d, K, m, n_samples, step_size, lambdas] = SetParameters(Xtrain, Ytrain);
rng(400);

lambdas_size = size(lambdas)




% Gradient test
% TestGradient(Xtrain, Ytrain, W, b);

% % Etas test
% [n_epochs, etas] = CyclicEtas(10000, 100, 2, 500, 0.05, 0.01);
% X = 1:length(etas);
% plot(etas)
% n_epochs


% % Sanity check 
% Xtrain_small = Xtrain(:, 1:100);
% Ytrain_small = Ytrain(:, 1:100);
% d = size(Xtrain, 1)
% K = size(Ytrain, 1)
% [W, b] = initWb(d, K, 50);
% costs = zeros(200);
% eta = 0.2
% n_layers = 2;
% for i=1:200
%     [Wgrad, bgrad] = ComputeGradients(Xtrain_small, Ytrain_small, W, b, 0);
%     for layer = 1:n_layers
%         W{layer} = W{layer} - eta*Wgrad{layer};
%         b{layer} = b{layer} - eta*bgrad{layer};
%     end
%     
%     costs(i) = ComputeCost(Xtrain_small, Ytrain_small, W, b, 0);
% end
% plot(costs)


% % Gradient descent
[Xtrain, Xval, Xtest] = Preprocess(Xtrain, Xval, Xtest);

batch_size = 100; n_cycles = 2; bottom = 1e-5; top = 1e-1;
[n_epochs, etas] = CyclicEtas(n_samples, batch_size, n_cycles, step_size, top, bottom);
plot(etas)

test_accuracies = zeros(1, size(lambdas, 2));
for i = 1:size(lambdas, 2)
    test_accuracies(i) = MiniBatchGD(Xtrain, Ytrain, ytrain, ...
        Xval, Yval, yval, ...
        Xtest, Ytest, ytest, ...
        batch_size, n_cycles, step_size, top, bottom, ...
        lambdas(i))
end

function [Xtrain, Ytrain, ytrain, ...
    Xval, Yval, yval, ...
    Xtest, Ytest, ytest] = BigLoad()
    
    [Xtrain, Ytrain, ytrain] = LoadBatch(1);
    for i = 2:4
        [X ,Y, y] = LoadBatch(i);
        Xtrain = [Xtrain, X];
        Ytrain = [Ytrain, Y];
        ytrain = [ytrain, y];
    end
    [X5, Y5, y5] = LoadBatch(5);
    Xtrain = [Xtrain, X5(:, 1:5000)];
    Ytrain = [Ytrain, Y5(:, 1:5000)];
    ytrain = [ytrain, y5(:, 1:5000)];
    
    Xval = X5(:, 5001:10000);
    Yval = Y5(:, 5001:10000);
    yval = y5(:, 5001:10000);
    
    [Xtest, Ytest, ytest] = LoadBatch(0);
end


function [d, K, m, n_samples, step_size, lambdas] = SetParameters(Xtrain, Ytrain)
    [d, n_samples] = size(Xtrain);
    batch_size = 100;
    K = size(Ytrain, 1);
    m = 50;
    step_size = 2 * floor(n_samples / batch_size);
    
    l_min = -5; l_max = -1;
    
    lambdas = zeros(1, 20);
    for i = 1 : 20
        l = l_min + (l_max - l_min) * rand(1, 1);
        lambdas(i) = 10^l;
    end
    
end

function  [Xtrain, Xval, Xtest] = Preprocess(Xtrain, Xval, Xtest)
    norm = mean(Xtrain, 2);
    Xtrain = Xtrain - norm;
    Xval = Xval - norm;
    Xtest = Xtest - norm;
end

function TestGradient(Xtrain, Ytrain, W, b)
lambda = 0;
[agrad_W, agrad_b] = ComputeGradients(Xtrain(:, 1), Ytrain(:, 1), W, b, lambda);
[ngrad_W, ngrad_b] = ComputeGradsNumSlow(Xtrain(:, 1), Ytrain(:, 1), W, b, lambda, 1e-6); 
norm(agrad_W{1} - ngrad_W{1})
norm(agrad_W{2} - ngrad_W{2})
size(agrad_b{1})
size(ngrad_b{1})
size(agrad_b{2})
size(ngrad_b{2})

norm(agrad_b{1} - ngrad_b{1})
norm(agrad_b{2} - ngrad_b{2})

end

function [W, b] = initWb(d, K, m)
    W{1} = double(1/sqrt(d) * randn(m, d));
    W{2} = double(1/sqrt(m) * randn(K, m));
    b{1} = double(zeros(m, 1));
    b{2} = double(zeros(K, 1));
end

function [loss, cost] = ComputeCost(X, Y, W, b, lambda)
    [~, P] = EvaluateClassifier(X, W, b);
    loss = 0;
    for i = 1:size(X, 2)
        loss = loss - log(Y(:, i)' * P(:, i));
    end
    loss = loss / size(X, 2);
    cost = loss + lambda * (sum(sum(W{1}.^2),2) + sum(sum(W{2}.^2),2));
end

function [X, Y, y] = LoadBatch(batch_number)
    addpath datasets/cifar-10-batches-mat/;
    if batch_number == 0
        filename = 'test_batch.mat';
    else
    
        filename = strcat('data_batch_', string(batch_number), '.mat')
    end

    file = load(filename);
    
    X = double(file.data)' / 255;
    y = (file.labels + 1)';
    Y = double(zeros(10, length(y)));
    for i = 1:length(y)
        Y(y(i),i) = 1;
    end

    
end

function [h, P] = EvaluateClassifier(X, W, b)
    s{1} = W{1} * X + b{1};
    h = max(0, s{1});
    s{2} = W{2} * h + b{2};
    P = softmax(s{2});
end

function [n_epochs, etas] = CyclicEtas(n_samples, batch_size, n_cycles, step_size, top, bottom)
    n_epochs = 2 * step_size * n_cycles * batch_size / n_samples;
    k = (top - bottom) / step_size;
    t = 1:step_size;
    half_cycle = bottom + k * t;
    full_cycle = [half_cycle, fliplr(half_cycle)];
    etas = [];
    for i = 1:n_cycles
        etas = [etas, full_cycle];
    end
end

function test_acc = MiniBatchGD(X, Y, y, ...
    Xval, Yval, yval, ...
    Xtest, Ytest, ytest, ...
    batch_size, n_cycles, step_size, top, bottom, ...
    lambda)
    n_layers = 2;  
    [d, n_samples] = size(X);
    K = size(Y, 1);
    [n_epochs, etas] = CyclicEtas(n_samples, batch_size, n_cycles, step_size, ...
        top, bottom);
    n_updates = n_samples * n_epochs / batch_size;
    
    [Wstar, bstar] = initWb(d, K, 50);
    
%     
%     
%     losses = zeros(1, n_updates);
%     costs = zeros(1, n_updates);
%     accuracies = zeros(1, n_updates);
%     for update = 1:n_updates
%         inds =  mod((update - 1) * batch_size + 1, n_samples) : mod(update * batch_size, n_samples);
%         Xbatch = X(:, inds);
%         Ybatch = Y(:, inds);
%         [Wgrad, bgrad] = ComputeGradients(Xbatch, Ybatch, Wstar, bstar, lambda);
%         eta = etas(update);
%         for layer = 1:n_layers
%             Wstar{layer} = Wstar{layer} - eta*Wgrad{layer};
%             bstar{layer} = bstar{layer} - eta*bgrad{layer};
%         end
%         [losses(update), costs(update)] = ComputeCost(X, Y, Wstar, bstar, lambda);
%         accuracies(update) = ComputeAccuracy(X, y, Wstar, bstar);
%     end
    

    
    costs = zeros(1, n_epochs);
    val_costs = zeros(1, n_epochs);
    losses = zeros(1, n_epochs);
    val_losses = zeros(1, n_epochs);
    accuracies = zeros(1, n_epochs);
    val_accuracies = zeros(1, n_epochs);
    t = 1;
    
    for epoch = 1:n_epochs
        for j = 1:n_samples / batch_size
            eta = etas(t);
            t = t + 1;
            inds = (j-1)*batch_size + 1 : j*batch_size;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            
            [Wgrad, bgrad] = ComputeGradients(Xbatch, Ybatch, Wstar, bstar, lambda);
            
            for layer = 1:n_layers
                Wstar{layer} = Wstar{layer} - eta*Wgrad{layer};
                bstar{layer} = bstar{layer} - eta*bgrad{layer};
            end
     
            
        end
        [losses(epoch), costs(epoch)] = ComputeCost(X, Y, Wstar, bstar, lambda);
        [val_losses(epoch), val_costs(epoch)] = ComputeCost(Xval, Yval, Wstar, bstar, lambda);
        accuracies(epoch) = ComputeAccuracy(X, y, Wstar, bstar);
        val_accuracies(epoch) = ComputeAccuracy(Xval, yval, Wstar, bstar);
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
    
    test_acc = ComputeAccuracy(Xtest, ytest, Wstar, bstar)
%     
%     print('~/Documents/MATLAB/Deep Learning/ResultPics/6cost_accuracy.pdf', '-dpdf', '-bestfit')
end

function acc = ComputeAccuracy(X, y, W, b)
    [~, P] = EvaluateClassifier(X, W, b);
    [~, argmax] = max(P);
    acc = double(sum(eq(argmax, y))/length(y));
end

function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
    n_samples = size(X, 2);
    [h, P] = EvaluateClassifier(X, W, b);
    G = (P-Y);
    grad_W{2} = G * h' / n_samples + 2 * lambda * W{2};
    grad_b{2} = G / n_samples * ones(n_samples, 1);
   
    G = W{2}' * G;
    G = G .* (h > 0);
    
    grad_b{1} = G / n_samples * ones(n_samples, 1);
    
    grad_W{1} = G * X' / n_samples + 2 * lambda * W{1};
end

function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W, b_try, lambda);
        
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W, b_try, lambda);
        
        grad_b{j}(i) = (c2-c1) / (2*h);
    end
end

for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) - h;
        c1 = ComputeCost(X, Y, W_try, b, lambda);
        
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        c2 = ComputeCost(X, Y, W_try, b, lambda);
        
        grad_W{j}(i) = (c2-c1) / (2*h);
    end
end
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
