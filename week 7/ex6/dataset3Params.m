function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% create two column vectors of parameter options to iterate through
C_vect = [0.01 0.03 0.1 0.3 1 3 10 30];
sig_vect = [0.01 0.03 0.1 0.3 1 3 10 30];

% create a result matrix of size 3x64 to append the [error; C; sigma;] results to
result_mat = ones(3,(length(C_vect) * length(sig_vect)));
% a helper index for appending the computed results to the result matrix
ind = 1;

% outer loop
for i = 1:length(C_vect);
	% inner loop
	for j = 1:length(sig_vect);
		% choose the parameters
		C = C_vect(i);
		sigma = sig_vect(j);
		
		% train the model with the given parameters
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
		% make predictions on the model with the Xval subset
		prediction = svmPredict(model, Xval);
		% compute the error between the prediction and the yval subset
		error = mean(double(prediction ~= yval));
		% append the result vector to the ind place in the matrix
		result_mat(:, ind) = [error; C; sigma;];
		
		% increment the ind
		ind = ind + 1;
	end;
end;

% find the index of the lowest error (first row, all columns)
[~, I] = min(result_mat(1,:));

% the vector of the I-th column in the matrix has the optimal parameters
best_params = result_mat(:, I);

% return those params
C = best_params(2);
sigma = best_params(3);

% =========================================================================

end
