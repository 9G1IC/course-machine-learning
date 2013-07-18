function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, ...
                                   num_labels, X, y, lambda)


% Setup some useful variables
k1 = m = size(X,1);
k2 = hidden_layer_size;
k3 = k = num_labels;
Y = eye(k)(y,:);

% Reshape
% =====================================================================
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));
T1_grad = zeros(size(Theta1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
T2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part1: Cost function with Feedforward
% ======================================================================

% FeedFoward
a1 = [ones(m,1) X];
z2 = Theta1*a1'; 
a2 = [ones(1,m); sigmoid(z2)]; 
z3 = Theta2*a2;               
a3 = sigmoid(z3);            

% Regolarization
t1 = Theta1(:,2:end);
t2 = Theta2(:,2:end);
reg = (lambda/(2*m))*( sumsq(t1(:)) + sumsq(t2(:)) );

% Cost Function
cost = Y.*log(a3)' + (1-Y).*log(1-a3)';
J = -(1/m)*sum( cost(:) ) + reg;


% Part2: Gradient with Backpropagation
% =========================================================================

for t = 1:m

    a1 = [1; X(t,:)'];
    z2 = Theta1 * a1;
    a2 = [1; sigmoid(z2)];

    z3 = Theta2 * a2;
    a3 = sigmoid(z3);

    yy = ([1:num_labels]==y(t))';
    delta_3 = a3 - yy;
    delta_2 = (Theta2' * delta_3) .* [1; sigmoidGradient(z2)];
    delta_2 = delta_2(2:end); 
                            
    T1_grad = T1_grad + delta_2 * a1';
    T2_grad = T2_grad + delta_3 * a2';
    end

% Regolarization

T1_grad = (1/m) * T1_grad + (lambda/m) * [zeros(size(Theta1, 1), 1) Theta1(:,2:end)];
T2_grad = (1/m) * T2_grad + (lambda/m) * [zeros(size(Theta2, 1), 1) Theta2(:,2:end)];

grad = [T1_grad(:); T2_grad(:)];

end
