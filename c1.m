clear all;
close all;
clc;

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2:end, :);

[y_m, y_n] = size(train_qso);
s_y = zeros(y_m, y_n);
x1 = lambdas;
X = [ones(size(x1, 1), 1), x1];
tau = 5;
max_iters = y_m;

for k = 1: max_iters
    for i = 1:y_n
        W = getWeight(x1(i), lambdas, tau);
        XtWx = X' * W * X;
        XtY = X' * W * (train_qso(k,:))';
        theta = XtWx \ XtY;
        s_y(k, i) = [1 x1(i)] * theta;
    end
end

save s_y


function  W = getWeight(x, x0, tau)
    m = size(x0, 1);
    W = zeros(m, m);
    for j = 1:m
        W(j, j) = exp(-(x - x0(j))^2 / (2 * tau^2));
    end
end