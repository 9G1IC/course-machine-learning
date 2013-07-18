function J = computeCostMulti(X, y, theta)

   J = norm( X*theta-y )^2 / (2*length(y));

end
