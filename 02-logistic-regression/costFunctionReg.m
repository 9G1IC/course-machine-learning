function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));

h = sigmoid(X * theta);

J =(1/m) * sum(-y.*log(h) - (1-y).*log(1-h)) + (lambda/(2*m)) * norm(theta([2:end]))^2;

gg = (lambda/m) .* theta;
gg(1) = 0;
   
grad = ((1/m)*X'*(h-y)) + gg;
   
end
