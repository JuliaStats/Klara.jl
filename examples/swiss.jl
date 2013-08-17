# Create design matrix X and response variable y from swiss data array
swiss = readdlm("swiss.txt", ' ');
covariates = swiss[:, 1:end-1];
ndata, npars = size(covariates);

covariates = (bsxfun(-, covariates, mean(covariates, 1))
  ./repmat(std(covariates, 1), ndata, 1));

polynomialOrder = 1;
X = zeros(ndata, npars*polynomialOrder);
for i = 1:polynomialOrder
  X[:, ((i-1)*npars+1):i*npars] = covariates.^i;
end

y = swiss[:, end];
