function p = predict(Theta1, Theta2, X)
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

A1 = [ones(m,1), X]; % (?x401)
A2 = [ones(m,1), sigmoid(A1*Theta1')]; % (?x401 * 401x25 = ?x25+1)
A3 = sigmoid(A2*Theta2'); % (?x25 * 26x10 = ?x10)

[vmax, p]  = max(A3,[],2);

end
