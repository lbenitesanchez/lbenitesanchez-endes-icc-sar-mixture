// Explicit experimental design-variance extension, not published generic TSLN.
// Same S2 priors and Gaussian marginalization; dynamic GVF includes log PSU count.
data {
  int<lower=1> M;
  int<lower=1> m;
  int<lower=1> P;
  int<lower=1> Q_gvf;
  matrix[M,P] Z;
  array[m] int<lower=1,upper=M> sampled_area;
  vector[m] theta_s1;
  vector<lower=0>[m] model_variance_s1;
  vector<lower=0>[m] sampling_sd_s1;
  int<lower=1,upper=m> stable_n;
  array[stable_n] int<lower=1,upper=m> stable;
  int<lower=0,upper=m> unstable_n;
  array[unstable_n] int<lower=1,upper=m> unstable;
  matrix[m,Q_gvf] L;
}
parameters {
  real intercept;
  vector[P] beta;
  real<lower=0> sigma_area;
  vector[Q_gvf] omega;
  real<lower=0> sigma_gvf;
}
transformed parameters {
  vector[M] linking_mean=intercept+Z*beta;
  vector[m] sampling_sd=sampling_sd_s1;
  vector[m] log_sampling_sd_mean=L*omega;
  vector[m] observation_variance;
  if(unstable_n>0)
    for(k in 1:unstable_n)
      sampling_sd[unstable[k]]=exp(log_sampling_sd_mean[unstable[k]]
                                   +.5*square(sigma_gvf));
  observation_variance=model_variance_s1+square(sampling_sd);
}
model {
  intercept~student_t(3,0,1);
  beta~normal(0,2);
  sigma_area~normal(0,2);
  omega~normal(0,2);
  sigma_gvf~cauchy(0,2);
  log(sampling_sd_s1[stable])~normal(log_sampling_sd_mean[stable],sigma_gvf);
  theta_s1~normal(linking_mean[sampled_area],
                   sqrt(observation_variance+square(sigma_area)));
}
generated quantities {
  vector[M] theta;
  vector[M] prevalence;
  for(i in 1:M)theta[i]=normal_rng(linking_mean[i],sigma_area);
  for(k in 1:m){
    real A=square(sigma_area);
    real D=observation_variance[k];
    real r=A/(A+D);
    theta[sampled_area[k]]=normal_rng(linking_mean[sampled_area[k]]
      +r*(theta_s1[k]-linking_mean[sampled_area[k]]),sqrt(A*D/(A+D)));
  }
  prevalence=inv_logit(theta);
}
