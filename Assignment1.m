
clear;
close all

%% Data Loading
directory = 'DataSets/cifar-10-batches-mat/';
[Xtrain, ...
    Ytrain, ...
    ytrain] = LoadBatch(fullfile(directory, 'data_batch_1.mat'));
[Xval, ...
    Yval, ...
    yval] = LoadBatch(fullfile(directory, 'data_batch_2.mat'));
[Xtest, ...
    Ytest, ...
    ytest] = LoadBatch(fullfile(directory, 'test_batch.mat'));

sigma = .01;
rng(400);
[W, b] = initWb(Xtrain, sigma);


%% Gradient test
% lambda = 0.1;
% grad_args = {Xtrain(1:20, 1), Ytrain(:, 1), W(:, 1:20), b, lambda};
% [agrad_W, agrad_b] = ComputeGradients(grad_args{:});
% [ngrad_W, ngrad_b] = ComputeGradsNumSlow(grad_args{:}, 1e-6); 
% [Wabs, Wrel] = Errors(agrad_W, ngrad_W)
% [babs, brel] = Errors(agrad_b, ngrad_b)


%% Minibatch Gradient Descent
% GDparams containing [batch size, eta, nepochs]
GDparams = [100, .01, 80];
lambda = 0.001;
[Wstar, bstar] = MiniBatchGD(Xtrain, Ytrain, ytrain, Xval, Yval, yval, ...
    Xtest, Ytest, ytest, ...
    GDparams, W, b, lambda);

for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2. 1, 3]);
end
close all
montage(s_im, 'size', [1,10])
print('~/Documents/MATLAB/Deep Learning/ResultPics/5class_templates.pdf', '-dpdf', '-bestfit')


%% Network
function P = EvaluateClassifier(X, W, b)
    P = softmax(W*X + b);
end
function [grad_W, grad_b] = ComputeGradients(X, Y, W, b, lambda)
    % Input:
    %  X images of size d dim dxn
    %  Y ground truth dim Kxn
    % Output:
    %  grad W is the gradient matrix of the cost J relative to W. dim: K×d.
    %  grad b is the gradient vector of the cost J relative to b. dim: K×1.
    P = EvaluateClassifier(X, W, b);
    n = size(X, 2);
    G = (P-Y);
    grad_b = G / n * ones(n,1);
    grad_W = G * X' / n + 2 * lambda * W;
end
function [grad_W, grad_b] = ComputeGradsNumSlow(X, Y, W, b, lambda, h)

no = size(W, 1);
d = size(X, 1);

grad_W = zeros(size(W));
grad_b = zeros(no, 1);

for i=1:length(b)
    b_try = b;
    b_try(i) = b_try(i) - h;
    [~, c1] = ComputeCost(X, Y, W, b_try, lambda);
    b_try = b;
    b_try(i) = b_try(i) + h;
    [~, c2] = ComputeCost(X, Y, W, b_try, lambda);
    grad_b(i) = (c2-c1) / (2*h);
end

for i=1:numel(W)
    
    W_try = W;
    W_try(i) = W_try(i) - h;
    [~, c1] = ComputeCost(X, Y, W_try, b, lambda);
    
    W_try = W;
    W_try(i) = W_try(i) + h;
    [~, c2] = ComputeCost(X, Y, W_try, b, lambda);
    
    grad_W(i) = (c2-c1) / (2*h);
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

%% Training
function [loss, cost] = ComputeCost(X, Y, W, b, lambda)
    P = EvaluateClassifier(X, W, b);
    loss = 0;
    for i = 1:size(X, 2)
        loss = loss - log(Y(:, i)' * P(:, i));
    end
    loss = loss / size(X, 2);
    cost = loss + lambda * (sum(sum(W.^2)));
end
function acc = ComputeAccuracy(X, y, W, b)
    P = EvaluateClassifier(X, W, b);
    [~, argmax] = max(P);
    acc = double(sum(eq(argmax, y)))/length(y);
end
function [Wstar, bstar] = MiniBatchGD(X, Y, y, ...
    Xval, Yval, yval, ...
    Xtest, Ytest, ytest, ...
    GDparams, W, b, lambda)
    N = size(X, 2);
    n_batch = GDparams(1);
    eta = GDparams(2);
    n_epochs = GDparams(3);
    Wstar = W;
    bstar = b;
    costs = zeros(1, n_epochs); val_costs = zeros(1, n_epochs);
    losses = zeros(1, n_epochs); val_losses = zeros(1, n_epochs);
    accuracies = zeros(1, n_epochs); val_accuracies = zeros(1, n_epochs);
    for epoch = 1:n_epochs
        for j = 1:N/n_batch
            inds = (j-1)*n_batch + 1:j*n_batch;
            Xbatch = X(:, inds);
            Ybatch = Y(:, inds);
            [Wgrad, bgrad] = ComputeGradients(Xbatch, Ybatch, Wstar, bstar, lambda);
            Wstar = Wstar - eta*Wgrad;
            bstar = bstar - eta*bgrad;
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
    
    print('~/Documents/MATLAB/Deep Learning/ResultPics/5cost_loss_accuracy.pdf', '-dpdf', '-bestfit')
    
    test_acc = ComputeAccuracy(Xtest, ytest, Wstar, bstar)

end

%% Parameters
function [W, b] = initWb(training_data, sigma)
    W = double(sigma * randn(10, size(training_data, 1)));
    b = double(sigma * randn(10, 1));
end

%% Data Loading
function [X, Y, y] = LoadBatch(filename)
    file = load(filename);
    X = double(file.data)' / 255;
    y = (file.labels + 1)';
    Y = double(zeros(10, length(y)));
    for i = 1:length(y)
        Y(y(i),i) = 1;
    end

end

%% Gradient test
function [abs, rel] = Errors(grad_a, grad_b)
    abs = norm(grad_a - grad_b);
    rel = abs / max(eps, norm(grad_a) + norm(grad_b));
end





