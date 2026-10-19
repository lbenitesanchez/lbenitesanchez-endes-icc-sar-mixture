functions {
  real gamma_increment_remainder(real logz, real c) {
    if (c == 0) return 0;
    if (logz < log(8.0)) {
      real z = exp(logz);
      return lgamma(z+c) - lgamma(1+z) + (1-c)*logz;
    } else {
      real inverse = exp(-logz);
      real t = c*inverse;
      real lt = log1p(t);
      real h;
      real result;
      array[8] real coefficients = {1.0/12, -1.0/360, 1.0/1260, -1.0/1680,
        1.0/1188, -691.0/360360, 1.0/156, -3617.0/122400};
      if (t < 0.01) {
        h = 0;
        for (step in 1:8) {
          int m = 9-step;
          real coefficient = (m % 2 == 1 ? -1.0 : 1.0)/(m+1);
          h = t*(h+coefficient);
        }
      } else {
        h = lt/t-1;
      }
      result = c*h+(c-0.5)*lt;
      for (j in 1:8) {
        int order = 2*j-1;
        result += coefficients[j]*pow(inverse,order)*expm1(-order*lt);
      }
      return result;
    }
  }

  real stable_beta_power(real eta, real log_kappa, real k, real f) {
    real log_mu = log_inv_logit(eta);
    real log_complement = log_inv_logit(-eta);
    return k*log_mu+f*log_complement
      +gamma_increment_remainder(log_kappa+log_mu,k)
      +gamma_increment_remainder(log_kappa+log_complement,f)
      -gamma_increment_remainder(log_kappa,k+f);
  }
}
