function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %


	% WORKING CODE
	
	% transpose 2x1 vector theta and 97x2 matrix X so that inner matrix is the same
	% commutativity is allowed here (matrix x vector multiplication)
	% result is 'a', 1x97 vector
	a = theta' * X';
	
	% transpose it so I can subtract vector y from it
	b = a' - y;
	
	% transpose the above result so I can multiply it with the original X matrix
	% new result is a 1x2 vector. Transpose it so that it's of the same dimension as the original theta
	c = (b' * X)';
	
	% store the result under newtheta
	newtheta = theta - alpha/m * c;
	
	% update theta with the newtheta result (simultaneous updating!)
	theta = newtheta;

	
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
