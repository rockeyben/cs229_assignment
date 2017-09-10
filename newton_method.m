clear all;
close all;
clc;

X = load('logistic_x.txt');
y = load('logistic_y.txt');
alpha = 0.0001;
X = [ones(size(X,1), 1) ,X];
[m, n] = size(X);

theta = sym('theta',[n,1])
for i = 1:n
    eval(['syms theta', num2str(i)]);
    eval(['theta(', num2str(i),')=theta',num2str(i)]);
end

max_iters = 10;
ll = (1/m) * sum(log(1+exp(-y.*(X*theta))));

for i = 1:n
    eval(['theta',num2str(i),'=',num2str(0)]);
end

thetaVal = zeros(n, 1)
digits(5);

for i = 1:max_iters
  for j = 1: n
     eval(['theta', num2str(j), '=thetaVal(', num2str(j), ', 1)']);
  end
  H = hessian(ll, theta);
  HH = vpa(subs(H));
  gra = vpa(subs(gradient(ll)));
  thetaVal = thetaVal - HH \ gra;
end

figure;
hold on;
plot(X(y < 0, 2), X(y < 0 ,3), 'rx' );
plot(X(y > 0, 2), X(y > 0 ,3), 'go' );

x1 = min(X(:, 2)) : .01 : max(X(:, 2));
x2 = (-thetaVal(1) / thetaVal(3)) - (thetaVal(2) / thetaVal(3)) * x1;
plot(x1, x2);
xlabel('x1');
ylabel('x2');


