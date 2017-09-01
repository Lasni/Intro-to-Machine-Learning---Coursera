function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



% compute h_theta(X) 
H = sigmoid(X * theta);

% create a new theta vector where the first element theta0 is always 0
new_theta = [0; theta(2:length(theta), :)];


% compute the regularization term
regTerm = lambda * sum(new_theta .^ 2) / (2 * m);

% parts of the cost function from costFunction.m
a = -y' * log(H);
b = 1 - y;
c = log(1 - H);

% added the regTerm to the original cost function from costFunction.m
J = sum(a - b' * c) / m + regTerm;

% added the regularization term
grad = (((H - y)' * X)' + lambda * new_theta) / m;

% =============================================================

end
