function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

m = length(y);
J_history = zeros(num_iters, 1);
 
for iter = 1:num_iters
    theta -= (alpha/m) *  X'*(X*theta - y);
    J_history(iter) = computeCostMulti(X, y, theta);
end

end
