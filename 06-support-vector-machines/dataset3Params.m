function [C, sigma] = dataset3Params(X, y, Xval, yval)

CTargets = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
SigmaTargets =[ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];

params = zeros(3,1);
for C = CTargets
	for Sigma = SigmaTargets
		model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, Sigma));
		prediction = svmPredict(model, Xval);
		predictedErr = mean(double(prediction == yval));
		if(predictedErr > params(1))
			params = [predictedErr, C, Sigma];
		end
	end
end

C = params(2);
sigma = params(3);

end
