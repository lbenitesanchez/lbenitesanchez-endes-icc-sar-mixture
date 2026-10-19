// New S2 spatial sensitivity following Hogg Supporting Information D.2.
// Author-case priors; no Australian benchmarking or S1 individual residual.
// IID latent effects and Gaussian measurement errors are integrated exactly.
// Every nonzero ICAR eigenmode is retained, with component-wise exact sum-zero.
// This is a full-rank constrained CAR whitening, not a low-rank spatial basis.
data {
  int<lower=2> M;
  int<lower=1> m;
  int<lower=1> P;
  matrix[M, P] Z;
  array[m] int<lower=1, upper=M> sampled_area;
  vector[m] theta_s1;
  vector<lower=0>[m] model_variance_s1;
  vector<lower=0>[m] sampling_sd_s1;
  int<lower=0, upper=m> stable_n;
  array[stable_n] int<lower=1, upper=m> stable;
  int<lower=0, upper=m> unstable_n;
  array[unstable_n] int<lower=1, upper=m> unstable;
  matrix[m, 2] L;
  int<lower=1> R;
  matrix[M,R] scaled_icar_factor;
}
transformed data {
  // Exact row extraction: unsampled predictions do not enter the likelihood.
  // Every CAR coordinate remains in the prior; the full factor is used below.
  matrix[m,R] sampled_icar_factor = scaled_icar_factor[sampled_area];
}
parameters {
  real intercept;
  vector[P] beta;
  real<lower=0> sigma_area;
  real<lower=0, upper=1> rho;
  vector[R] spatial_z;
  vector[2] omega;
  real<lower=0> sigma_gvf;
}
transformed parameters {
  vector[m] linking_mean_sampled;
  vector[m] sampling_sd = sampling_sd_s1;
  vector[m] log_sampling_sd_mean = L * omega;
  vector[m] observation_variance;
  real iid_variance = square(sigma_area) * (1-rho);
  linking_mean_sampled = intercept + Z[sampled_area] * beta
      + sigma_area * sqrt(rho) * (sampled_icar_factor * spatial_z);
  if (unstable_n > 0)
    for (k in 1:unstable_n)
      sampling_sd[unstable[k]] = exp(log_sampling_sd_mean[unstable[k]]
                                     + .5*square(sigma_gvf));
  observation_variance = model_variance_s1 + square(sampling_sd);
}
model {
  intercept ~ student_t(3,0,1);
  beta ~ normal(0,2);
  sigma_area ~ normal(0,2);
  rho ~ beta(3.05,1.65);
  spatial_z ~ std_normal();
  omega ~ normal(0,2);
  sigma_gvf ~ cauchy(0,2);
  log(sampling_sd_s1[stable]) ~ normal(log_sampling_sd_mean[stable],sigma_gvf);
  theta_s1 ~ normal(linking_mean_sampled,
                    sqrt(observation_variance+iid_variance));
}
generated quantities {
  vector[M] linking_mean = intercept + Z * beta
      + sigma_area * sqrt(rho) * (scaled_icar_factor * spatial_z);
  vector[M] theta;
  vector[M] prevalence;
  for (i in 1:M) theta[i]=normal_rng(linking_mean[i],sqrt(iid_variance));
  for (k in 1:m) {
    real D=observation_variance[k];
    real w=iid_variance/(iid_variance+D);
    real mu=linking_mean[sampled_area[k]]
             + w*(theta_s1[k]-linking_mean[sampled_area[k]]);
    theta[sampled_area[k]]=normal_rng(mu,sqrt(iid_variance*D/(iid_variance+D)));
  }
  prevalence=inv_logit(theta);
}
