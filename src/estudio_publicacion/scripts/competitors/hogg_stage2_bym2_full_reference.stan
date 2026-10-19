// New S2 spatial sensitivity following Hogg Supporting Information D.2.
// Author-case priors; no Australian benchmarking or S1 individual residual.
// IID latent effects and Gaussian measurement errors are integrated exactly.
// Component-wise ICAR sum-zero constraints use an orthonormal Householder map.
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
  int<lower=1> C;
  array[C] int<lower=2> component_size;
  array[M] int<lower=1, upper=M> node_order;
  vector<lower=0>[M] icar_scale;
  int<lower=1> E;
  array[E] int<lower=1, upper=M> node1;
  array[E] int<lower=1, upper=M> node2;
}
parameters {
  real intercept;
  vector[P] beta;
  real<lower=0> sigma_area;
  real<lower=0, upper=1> rho;
  vector[M-C] spatial_free;
  vector[2] omega;
  real<lower=0> sigma_gvf;
}
transformed parameters {
  vector[M] spatial_raw;
  vector[M] spatial_effect;
  vector[M] linking_mean;
  vector[m] sampling_sd = sampling_sd_s1;
  vector[m] log_sampling_sd_mean = L * omega;
  vector[m] observation_variance;
  real iid_variance = square(sigma_area) * (1-rho);
  {
    int node_start=1;
    int free_start=1;
    for (c in 1:C) {
      int n=component_size[c];
      real sr=sum(segment(spatial_free,free_start,n-1));
      real root_n=sqrt(1.0*n);
      for (j in 1:(n-1))
        spatial_raw[node_order[node_start+j-1]] = spatial_free[free_start+j-1]
                                           - sr/(root_n*(root_n-1));
      spatial_raw[node_order[node_start+n-1]] = sr/root_n;
      node_start += n;
      free_start += n-1;
    }
  }
  spatial_effect = sigma_area * sqrt(rho) * spatial_raw ./ sqrt(icar_scale);
  linking_mean = intercept + Z * beta + spatial_effect;
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
  omega ~ normal(0,2);
  sigma_gvf ~ cauchy(0,2);
  target += -.5*dot_self(spatial_raw[node1]-spatial_raw[node2]);
  log(sampling_sd_s1[stable]) ~ normal(log_sampling_sd_mean[stable],sigma_gvf);
  theta_s1 ~ normal(linking_mean[sampled_area],
                    sqrt(observation_variance+iid_variance));
}
generated quantities {
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
