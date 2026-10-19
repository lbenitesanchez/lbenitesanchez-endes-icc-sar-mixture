#include beta_power_stable_functions.stan
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
model {
  vector[A] eta=X*beta+B*gamma;
  beta[1] ~ normal(0,intercept_sd);
  if(P>1) beta[2:P] ~ normal(0,beta_sd);
  gamma ~ normal(0,spatial_sd);
  log_kappa ~ normal(log_kappa_mean,log_kappa_sd);
  // Computational branch only; the precision prior/support is unchanged.
  // Large precision and extreme logits always use the stable increment target.
  if (log_kappa > -6 && log_kappa < 8 && max(abs(eta[idx])) < 30) {
    real kappa = exp(log_kappa);
    vector[S] a = kappa*inv_logit(eta[idx]);
    vector[S] b = kappa*inv_logit(-eta[idx]);
    target += sum(lbeta(a+successes,b+failures)-lbeta(a,b));
  } else {
    for(s in 1:S)
      target += stable_beta_power(eta[idx[s]],log_kappa,successes[s],failures[s]);
  }
}
