clear all;
close all;
clc;

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2, :)';
figure;
hold on;
h = plot(lambdas', train_qso, 'k+');
set(h, 'linewidth', 1);
x1 = lambdas;
size(x1)
q_n = size(x1, 1)
X = [ones(q_n, 1), x1];
tau = [1, 5, 10, 100, 1000];
colors=['r-', 'b-', 'g-', 'm-', 'c-'];
iters = size(tau, 2);
for k = 1:iters
    x2 = zeros(1, q_n);
for i = 1: q_n
    W = getWeight(x1(i), lambdas, tau(k));
    theta = inv(X' * W * X) * X' * W * train_qso;
    x2(i) = [1 x1(i)] * theta;
end
h = plot(x1, x2, char(colors(k)));
set(h, 'linewidth', 2);
end

h = legend('Raw data', 'tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000');
set(h, 'fontsize', 10);


function  W = getWeight(x, x0, tau)
    m = size(x0, 1);
    W = zeros(m, m);
    for j = 1:m
        W(j, j) = exp(-(x - x0(j))^2 / (2 * tau^2));
    end
end