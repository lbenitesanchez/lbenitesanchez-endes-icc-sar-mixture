data {
  int<lower=1> A;
  int<lower=1> P;
  int<lower=0> Q;
  matrix[A,P] X;
  matrix[A,Q] B;
  int<lower=1> S;
  array[S] int<lower=1,upper=A> idx;
  vector<lower=0>[S] successes;
  vector<lower=0>[S] failures;
  real<lower=0> beta_sd;
  real<lower=0> intercept_sd;
  real<lower=0> spatial_sd;
  real log_kappa_mean;
  real<lower=0> log_kappa_sd;
}
parameters {
  vector[P] beta;
  vector[Q] gamma;
  real log_kappa;
}
transformed parameters {
  vector[A] eta=X*beta+B*gamma;
}
model {
  real kappa=exp(log_kappa);
  beta[1] ~ normal(0,intercept_sd);
  if(P>1) beta[2:P] ~ normal(0,beta_sd);
  gamma ~ normal(0,spatial_sd);
  log_kappa ~ normal(log_kappa_mean,log_kappa_sd);
  for(s in 1:S) {
    real a=kappa*inv_logit(eta[idx[s]]);
    real b=kappa*inv_logit(-eta[idx[s]]);
    target += lbeta(a+successes[s],b+failures[s])-lbeta(a,b);
  }
}
