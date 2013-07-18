function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

delta = X*theta - y;
reg = (lambda/(2*m))*sumsq(theta(2:end,:));
J = (1/(2*m)) * sumsq(delta) + reg;

grad =  (1/m)*(X'*delta) + (lambda/m)*[0; theta(2:end,:)];

end
