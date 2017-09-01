function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% add the bias column of ones
X = [ones(m,1) X];
% assign X to A1 (formulaic clarity)
A1 = X;
% Z(j) = A(j-1) * Theta(j-1) (transpose if necessary)
Z2 = A1 * Theta1';
% A(j) = g(Z(j))
A2 = sigmoid(Z2);

% repeat as many times as needed, paying attention to the bias column and the sizes of matrices
A2 = [ones(m,1) A2];
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);

% get the biggest sigmoid value and its index for each example
[max_val, index] = max(A3, [], 2);

% assign index to p
p = index;



% =========================================================================


end
