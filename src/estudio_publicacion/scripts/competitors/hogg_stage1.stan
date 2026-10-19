// Hogg et al. (2024), generic TSLN stage 1, independently implemented.
// Collapsing identical (area, covariate) rows is an exact sufficient-statistic
// reduction of the weighted Bernoulli likelihood, not a binomial approximation.
data {
  int<lower=1> G;
  int<lower=1> m;
  int<lower=1> P;
  matrix[G, P] X;
  array[G] int<lower=1, upper=m> area;
  vector<lower=0>[G] weight_success;
  vector<lower=0>[G] weight_failure;
}
parameters {
  real intercept;
  vector[P] beta;
  vector[m] z_area;
  real<lower=0> sigma_area;
}
transformed parameters {
  vector[m] effect_area = sigma_area * z_area;
  vector[G] eta = intercept + X * beta + effect_area[area];
}
model {
  intercept ~ student_t(3, 0, 1);
  beta ~ normal(0, 2);
  z_area ~ std_normal();
  sigma_area ~ normal(0, 1);
  target += dot_product(weight_success, log_inv_logit(eta));
  target += dot_product(weight_failure, log_inv_logit(-eta));
}
