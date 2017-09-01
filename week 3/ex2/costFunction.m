function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% compute h_theta(X) 
H = sigmoid(X * theta);

% compute the cost function

% transpose the -y(100x1) so that inner dimensions agree and it can be multiplied with log(h)(100x3)
a = -y' * log(H);

% second part 
b = 1 - y;

% third part
c = log(1 - H);

% transpose b(100x1) so that inner dimensions agree and it can be multiplied with c (100x3)
J = sum(a - b' * c) / m;


% compute gradient
grad = ((H - y)' * X)' / m;



% =============================================================

end
