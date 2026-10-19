// Uncollapsed latent reference for independent Gaussian marginalization checks.
// Only the separately proven exact powered-draw reduction is applied here.
data {
  int<lower=1> M;
  int<lower=1> m;
  int<lower=1> P;
  int<lower=1> T;
  matrix[M, P] Z;
  array[m] int<lower=1, upper=M> sampled_area;
  matrix[T, m] theta_s1_draws;
  vector<lower=0>[m] model_variance_s1;
  vector<lower=0>[m] sampling_sd_s1;
  int<lower=0, upper=m> stable_n;
  array[stable_n] int<lower=1, upper=m> stable;
  int<lower=0, upper=m> unstable_n;
  array[unstable_n] int<lower=1, upper=m> unstable;
  matrix[m, 2] L;
}
transformed data {
  vector[m] theta_s1_mean;
  for (k in 1:m) theta_s1_mean[k] = mean(theta_s1_draws[, k]);
}
parameters {
  real intercept;
  vector[P] beta;
  real<lower=0> sigma_area;
  vector[M] z_area;
  vector[m] bar_theta;
  vector[2] omega;
  real<lower=0> sigma_gvf;
}
transformed parameters {
  vector[M] theta = intercept + Z * beta + sigma_area * z_area;
  vector[M] prevalence = inv_logit(theta);
  vector[m] sampling_sd = sampling_sd_s1;
  vector[m] log_sampling_sd_mean = L * omega;
  if (unstable_n > 0)
    for (k in 1:unstable_n)
      sampling_sd[unstable[k]] = exp(log_sampling_sd_mean[unstable[k]]
                                    + 0.5 * square(sigma_gvf));
}
model {
  intercept ~ student_t(3, 0, 1);
  beta ~ normal(0, 2);
  sigma_area ~ normal(0, 2);
  z_area ~ std_normal();
  omega ~ normal(0, 2);
  sigma_gvf ~ cauchy(0, 2);
  log(sampling_sd_s1[stable]) ~ normal(log_sampling_sd_mean[stable], sigma_gvf);
  for (k in 1:m) {
    target += normal_lpdf(theta_s1_mean[k] | bar_theta[k],
                         sqrt(model_variance_s1[k]));
    bar_theta[k] ~ normal(theta[sampled_area[k]], sampling_sd[k]);
  }
}
