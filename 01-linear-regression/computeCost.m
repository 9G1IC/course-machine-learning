function J = computeCost(X, y, theta)
m = length(y); % number of training examples

s = 0;

for i=1:m,
    s += (theta(1) + theta(2) * X(i,2) - y(i))^2;

J = s/(2*m);

end
