clear all;
close all;
clc;

load quasar_train.csv;
lambdas = quasar_train(1, :)';
train_qso = quasar_train(2, :)';
figure;
hold on;
plot(lambdas', train_qso, 'rx')
lambdas = [ones(size(lambdas, 1), 1), lambdas];

theta = inv(lambdas' * lambdas) * lambdas' * train_qso
x1 = min(lambdas(: , 2)): 1 : max(lambdas(:, 2));
x2 = theta(2) .* x1 + theta(1);

plot(x1, x2)