# Create design matrix X and response variable y from vaso data array
vaso = readdlm("vaso.txt", ' ');
covariates = vaso[:, 1:end-1];
ndata, npars = size(covariates);

covariates = (bsxfun(-, covariates, mean(covariates, 1))
  ./repmat(std(covariates, 1), ndata, 1));

polynomialOrder = 1;
X = ones(ndata, npars*polynomialOrder+1);
for i = 1:polynomialOrder
  X[:, ((i-1)*npars+2):(i*npars+1)] = covariates.^i;
end
npars += 1;

y = vaso[:, end];
