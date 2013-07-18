function [J, grad] = lrCostFunction(theta, X, y, lambda)
m = length(y); % number of training examples
t2 = theta(2:end);

h = sigmoid(X*theta);

J = -(y'*log(h) + (1-y)'*log(1-h))/m;
J += (lambda/(2*m))*(t2'*t2);

grad = (X'*(h-y))/m;
grad(2:end) += (lambda/m)*t2;

end
