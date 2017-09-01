% the function is going to be calculating the training error and cross validation error
% from multiple sets of randomly selected examples and then displaying both errors 
% relative to the number of training examples in a plot

function randomPlot(X, y, Xval, yval)
	
	% number of training examples
	m = size(X, 1);
	
	% response vectors used for storing our error values
	train_err = zeros(m, 1);
	val_err = zeros(m, 1);
	
	% optimal lambda value
	lambda = 3;
	
	% number of times (outer loop) we're gonna evaluate theta and compute the error on the
	% same specific set, for all random sets of sizes 1 through m (inner loop)
	o_loop = 50;
	
	
	% do it o_loop times 
	for loop = 1:o_loop;
		
		% for sizes from 1 to m
		for i = 1:m;
		
			% randomly shuffle the numbers that will be used for selecting the rows in X and y
			shuffle = randperm(m);
			
			% select i of these random numbers
			selected = shuffle(1:i);
			
			% use them to extract the corresponding rows from X and y
			X_sub = X(selected, :);
			y_sub = y(selected, :);
			
			% learn the theta parameters using these randomly chosen subsets of size i
			theta = trainLinearReg(X_sub, y_sub, lambda);
			
			% evaluate the theta parameters on this same set (without regularization - lambda=0)
			[J, grad] = linearRegCostFunction(X_sub, y_sub, theta, 0);
			
			% accumulate the training errors
			train_err(i) = train_err(i) + J;
			
			
			% repeat the above procedure for the cross-validation set
			shuffle = randperm(size(Xval, 1));
			selected = shuffle(1:i);
			X_sub = Xval(selected, :);
			y_sub = yval(selected, :);
			[J, grad_val] = linearRegCostFunction(X_sub, y_sub, theta, 0);
			val_err(i) = val_err(i) + J;
		end
	end
	
	% average the errors
	train_err = train_err ./ o_loop;
	val_err = val_err ./ o_loop;
	
	% plot the graph
	plot(1:m, train_err, 1:m, val_err);
	xlabel('Number of training examples');
	ylabel('Error');
	axis([0 13 0 100]);
	legend('Training set', 'Cross-Validation set');
end