#set document(title: "Background and Notation")
#set page(paper: "a4", margin: 2.5cm, numbering: "1")
#set text(
  font: ("Libertinus Serif", "New Computer Modern", "Noto Serif"),
  size: 10.5pt,
  lang: "en",
)
#set par(justify: true, leading: 0.65em)
#set heading(numbering: none)
#show heading: set block(above: 1.6em, below: 0.9em)
#show raw.where(block: true): set block(
  width: 100%,
  inset: (x: 1em, y: 0.8em),
  fill: luma(247),
)

= Background and Notation

Empirical dynamic modelling represents an unobserved dynamical state by a delay vector and learns relationships between delay vectors from observed data.
This chapter defines the state representation, the relationship to be learned and the three estimators used to learn it.

== 1. The data

We consider a *multivariate time series*, a sequence of observations of $d$ variables sampled at a fixed interval,

$
  x_t = (x_t^((1)), dots, x_t^((d))) in bb(R)^d,
  quad t = 0, 1, dots, T-1
$

where $T$ is the length of the series and $d$ the number of variables.

Behind the observations, we assume a deterministic dynamical system with a state $z_t$ that is not observed directly.
The state evolves according to a time-independent dynamics $Phi$,

$ z_(t+1) = Phi(z_t) $

and each observed variable is a function of the state,

$ x_t^((i)) = g_i (z_t) $

Let $A$ be a compact set that is invariant under $Phi$, so $Phi(A) = A$.
We assume that $Phi$ is a diffeomorphism on an open neighbourhood of $A$.
For each $i in {1, dots, d}$, the observation function $g_i$ is smooth on that neighbourhood.

== 2. Delay coordinates

The state $z_t$ is hidden, so we instead represent it by observations at selected variables and lags.
We call the ordered sequence of selected (variable index, lag) pairs the *delay coordinates*,

$
  cal(J) = ( (i_1, ell_1), dots, (i_E, ell_E) ),
  quad i_j in {1, dots, d}, quad ell_j in bb(Z)
$

Each pair $(i_j, ell_j)$ is one delay coordinate.
The number $E = |cal(J)|$ is the dimension of the resulting delay vector.
Evaluating $x$ at the selected delay coordinates gives the *delay vector* at time $t$,

$ u_cal(J) (t) = ( x^((i_1))_(t - ell_1), dots, x^((i_E))_(t - ell_E) ) in bb(R)^E $

and its domain is the set of times at which all coordinates are defined,

$
  op("dom") u_cal(J) = { t in bb(Z) : 0 <= t - ell_j <= T-1 quad "for all" j } = { t in bb(Z) : max_j ell_j <= t <= T-1 + min_j ell_j }
$

Taking one variable at evenly spaced lags gives the classical univariate delay coordinates,

$ cal(J) = ( (1, 0), (1, tau), dots, (1, (E-1) tau) ) $

with lag $tau$.
Allowing $i_j$ to vary gives multivariate delay coordinates.
A negative lag $ell_j$ selects an observation taken $abs(ell_j)$ steps after time $t$.

Every entry of the delay vector $x^((i_j))_(t - ell_j)$ can be determined by the state at the single time $t$.
It is the observation of variable $i_j$ at time $t - ell_j$,

$ x^((i_j))_(t - ell_j) = g_(i_j) (z_(t - ell_j)) $

and invertibility ($Phi$ is a diffeomorphism) makes the forward or inverse iterate of $Phi$ well defined,

$ z_(t - ell_j) = Phi^(-ell_j) (z_t) $

Composing the observation function with the dynamics, the entry can be written as a function of $z_t$,

$ x^((i_j))_(t - ell_j) = g_(i_j) ( Phi^(-ell_j) (z_t) ) $

Collecting these entries as functions of the state defines the *delay-coordinate map* $Psi_cal(J)$,

$ Psi_cal(J) (z) = ( g_(i_1)(Phi^(-ell_1)(z)), dots, g_(i_E)(Phi^(-ell_E)(z)) ) $

The observed delay vector is the image of the current state under this map,

$ u_cal(J) (t) = Psi_cal(J) (z_t) $

The formula for $Psi_cal(J)$ can be constructed for any choice of delay coordinates $cal(J)$.
The remaining question is which choices make $u_cal(J)(t)$ a valid replacement for the unobserved state $z_t$.
The replacement retains the state only if distinct states have distinct delay vectors.
In terms of the delay-coordinate map, this condition is

$ Psi_cal(J)(z) = Psi_cal(J)(z') quad => quad z = z' $

for all $z, z' in A$.
It says that $Psi_cal(J)$ is one-to-one.
The inverse $Psi_cal(J)^(-1)$ can then recover the state from its delay vector.

A coordinate representation must also preserve nearness.
This requires $Psi_cal(J)^(-1)$ to be continuous on $Psi_cal(J)(A)$, so that nearby delay vectors correspond to nearby states.
A continuous one-to-one map whose inverse on its image is continuous is an *embedding*.
Because $A$ is compact and $Psi_cal(J)$ is continuous, one-to-one-ness is sufficient here: its inverse on $Psi_cal(J)(A)$ is then automatically continuous.

Embedding theorems connect this required property to the choice of delay coordinates.
They give sufficient conditions on their number and form under which $Psi_cal(J)$ is an embedding.
Takens proved that $2m+1$ consecutive delay coordinates produce an embedding of a compact $m$-dimensional manifold for generic smooth dynamics and a generic smooth scalar observation.
Sauer, Yorke and Casdagli replaced manifold dimension by box-counting dimension $D$ for compact invariant sets: under their regularity and periodic-orbit conditions, more than $2D$ coordinates suffice for prevalent observation maps.
Deyle and Sugihara extended the delay-coordinate result to multiple observation functions and non-consecutive lags#footnote[Takens (1981), “Detecting strange attractors in turbulence,” _Dynamical Systems and Turbulence, Warwick 1980_, Lecture Notes in Mathematics 898:366--381. Sauer, Yorke & Casdagli (1991), “Embedology,” _Journal of Statistical Physics_ 65(3--4):579--616. Deyle & Sugihara (2011), “Generalized theorems for nonlinear state space reconstruction,” _PLoS ONE_ 6(3):e18295.].

These results make a statement about typical observation functions, not every observation function.
*Generic* expresses typicality in a topological sense, while *prevalent* expresses it in a measure-theoretic sense.
Both allow exceptional functions for which the delay-coordinate map is not one-to-one.
The results also assume properties of the dynamics, including restrictions on short periodic orbits.
An observed dataset supplies particular functions $g_i$ and a particular map $Psi_cal(J)$.
The inequality $E > 2D$ therefore explains why an embedding can be expected under the theorem's assumptions, but it does not test whether the map constructed from that dataset is an embedding#footnote[A measure-theoretic variant fixes an invariant probability measure and replaces global injectivity by injectivity on a set of full measure. Under its stated regularity and periodic-point conditions, it needs more coordinates than the Hausdorff dimension of that measure: Barański, Gutman & Śpiewak (2020), “A probabilistic Takens theorem,” _Nonlinearity_ 33(9):4940--4966.].

== 3. The regression function

Choose delay coordinates $cal(J)$ of dimension $E$ and delay coordinates $cal(K)$ of dimension $E'$ from observations governed by the same state $z in A$.
They define two representations of that state,

$
  Psi_cal(J): A arrow.r bb(R)^E,
  quad Psi_cal(K): A arrow.r bb(R)^(E')
$

Suppose first that $Psi_cal(J)$ embeds $A$.
Its inverse recovers the state represented by any $q in Psi_cal(J)(A)$.
Applying $Psi_cal(K)$ to that state gives the induced map

$
  Psi_cal(K) compose Psi_cal(J)^(-1):
  Psi_cal(J)(A) arrow.r Psi_cal(K)(A)
$

The paired delay vectors then satisfy

$ u_cal(K)(t) = (Psi_cal(K) compose Psi_cal(J)^(-1))(u_cal(J)(t)) $

This composition is the reason to relate the two delay vectors.
It describes the $cal(K)$-vector directly from the $cal(J)$-vector, without using the unobserved state as an argument.
Because $Psi_cal(J)$ is an embedding, replacing $z_t$ with $u_cal(J)(t)$ discards none of the state information.

The construction depends on $Psi_cal(J)$ being an embedding.
The embedding theorems do not guarantee this property for the particular delay coordinates and observation functions in a dataset.
If $Psi_cal(J)$ is not one-to-one, its inverse is unavailable: the same $cal(J)$-vector can represent several states, and those states can have different $cal(K)$-vectors.
Measurement noise and stochastic dynamics can also make several $cal(K)$-vectors compatible with the same value of $q$.
The exact composition must therefore be replaced by a definition that remains meaningful when the relationship is one-to-many.

To describe that relationship probabilistically, let $mu$ be an invariant probability measure on $A$.
When $z_0$ has distribution $mu$, invariance makes the joint distribution of $u_cal(J)(t)$ and $u_cal(K)(t)$ independent of $t$.
A model for observation noise or stochastic dynamics supplies any additional randomness.
If $mu$ is ergodic under $Phi$, averages along almost every single trajectory converge to expectations under $mu$#footnote[Walters (1982), _An Introduction to Ergodic Theory_, Graduate Texts in Mathematics 79, Springer, states the invariant-measure and ergodic framework used here.].

A *regression function* assigns one representative value to the conditional distribution of the $cal(K)$-vector at each $q$.
The representative depends on the loss used to measure error.
For squared error, the unique optimal representative is the conditional mean,

$ F(q) = bb(E)[thin u_cal(K) (t) | u_cal(J) (t) = q thin] $

The relevant domain is the support of the $u_cal(J)(t)$ distribution, and each value $F(q)$ lies in $bb(R)^(E')$.
Because a conditional mean is an average, it need not equal any individual $cal(K)$-vector.
Among all measurable functions of $u_cal(J)(t)$, $F$ uniquely minimizes mean squared prediction error up to values on sets of probability zero#footnote[Györfi, Kohler, Krzyżak & Walk (2002), _A Distribution-Free Theory of Nonparametric Regression_, Springer Series in Statistics, chapter 1, develops regression around the conditional mean.].

For a continuously distributed $u_cal(J)(t)$, the event $u_cal(J)(t) = q$ usually has probability zero.
The conditional expectation is defined through a regular conditional distribution and does not require the exact vector $q$ to recur in the observed series.
The estimators below approximate $F(q)$ by combining $cal(K)$-vectors paired with $cal(J)$-vectors near $q$.

When $Psi_cal(J)$ embeds $A$ in the deterministic model, the conditional distribution is concentrated at $Psi_cal(K)(Psi_cal(J)^(-1)(q))$.
The regression function then reduces to the exact composition defined above.
It is continuous because $Psi_cal(K)$ and $Psi_cal(J)^(-1)$ are continuous.
For a non-injective reconstruction, noisy observations or stochastic dynamics, continuity of the conditional mean is an additional modelling assumption.
All three estimators rely on its local form: nearby values of $q$ have nearby conditional means.

Forecasting is encoded in $cal(K)$.
Replacing a lag $ell$ in $cal(K)$ by $ell-h$ moves that coordinate $h$ sampling steps forward.
The resulting regression function is a direct $h$-step forecast.
In a chaotic deterministic system, derivatives of the $h$-step state map can grow with $h$ along unstable directions#footnote[Eckmann & Ruelle (1985), “Ergodic theory of chaos and strange attractors,” _Reviews of Modern Physics_ 57(3):617--656, reviews characteristic exponents as rates of sensitivity to initial conditions.].
That growth can make the forecast function vary more rapidly and can reduce forecast skill, although the effect depends on the observation function and the region of state space.

== 4. The three matrices: $X$, $Y$ and $Q$

Estimating $F$ from finite data requires two sequences of times.
In this estimation problem, a $cal(J)$-vector serves as an input delay vector and its paired $cal(K)$-vector serves as an output delay vector.
The *library times* $L = (t_1, dots, t_N)$ are times at which both $u_cal(J)$ and $u_cal(K)$ can be read.
The *query times* $S = (s_1, dots, s_M)$ are times at which $u_cal(J)$ can be read and $u_cal(K)$ is to be estimated.

Library times require both delay vectors, while query times require only the input delay vector.

$
  L subset.eq op("dom") u_cal(J) inter op("dom") u_cal(K), quad
  S subset.eq op("dom") u_cal(J)
$

For any sequence of valid times $(r_1, dots, r_n)$, stack the corresponding delay vectors into the matrix

$ U_cal(J)((r_1, dots, r_n)) = mat(u_cal(J)(r_1); dots.v; u_cal(J)(r_n)) in bb(R)^(n times E) $

For example, let the single observed variable be $x_t = t$ for $t = 0, 1, dots, 9$, and take

$ cal(J) = ((1, 0), (1, 2), (1, 4)) $

Then $op("dom") u_cal(J) = {4, 5, dots, 9}$ and

$
  U_cal(J)((4, 5, dots, 9)) = mat(
    4, 2, 0;
    5, 3, 1;
    6, 4, 2;
    7, 5, 3;
    8, 6, 4;
    9, 7, 5
  )
$

Applying $u_cal(J)$ and $u_cal(K)$ to these two sequences gives three matrices.

$ X = U_cal(J) (L), quad Y = U_cal(K) (L), quad Q = U_cal(J) (S) $

Each row corresponds to one time and each column to one delay coordinate.
$X$ contains input delay vectors at the library times, $Y$ contains output delay vectors at those same times, and $Q$ contains input delay vectors at the query times.

#block(breakable: false)[
  #table(
    columns: 4,
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Matrix*], [*Size*], [*One row per*], [*One column per*]),
    [$X$], [$N times E$], [library time], [input coordinate],
    [$Y$], [$N times E'$], [the same library time], [output coordinate],
    [$Q$], [$M times E$], [query time], [input coordinate],
  )
]

The four objects $cal(J)$, $cal(K)$, $L$ and $S$ encode four parts of the estimation problem.

- $cal(J)$ selects the variables and lags used as input coordinates.
- $cal(K)$ selects the variables and lags used as output coordinates, including the forecast horizon.
- $L$ selects the paired observations used for fitting.
- $S$ selects the input delay vectors at which predictions are requested.


From here on, $q$ denotes a single query point, one row of $Q$, and everything is written for that one point.
The notation applies row by row even when several predictions share one fitted function or one set of tuning parameters.

== 5. Estimating $F$ from nearby observations

Each library row is one observation from the conditional distribution whose mean is $F$.
It can be written as

$ Y_n = F(X_n) + epsilon_n, quad bb(E)[epsilon_n | X_n] = 0 $

Continuity connects a query point to these observations.
When $X_n$ is close to $q$, $F(X_n)$ is close to $F(q)$.
The residual $epsilon_n$ remains, so several observations must be combined to reduce its effect.

Four decisions determine that combination.
The distance rule determines which input delay vectors count as similar.
The loss weights determine how strongly observations at different distances affect the fit.
The fitted function class determines which variation of $F$ can be represented near the query.
Regularization determines how strongly unsupported variation is suppressed.

Simplex projection and S-map express similarity as query-dependent loss weights.
GP-EDM uses equal loss weights and expresses similarity through a kernel in the fitted function class.
All three can be written as regularized weighted least squares,

$
  hat(f)_q = op("arg min", limits: #true)_(f in cal(F)) thin
  sum_(n=1)^N w_n (q) thin norm(Y_n - f(X_n))^2 + lambda thin Omega(f),
  quad hat(F)(q) = hat(f)_q (q)
$

Here $X_n$ and $Y_n$ are row $n$ of $X$ and $Y$.
The symbol $cal(F)$ denotes the fitted function class, $w_n(q)$ the loss weight, and $lambda Omega(f)$ the regularization penalty.
Query-dependent weights produce a separate fitted function $hat(f)_q$ for each $q$.
Equal weights produce one fitted function that is evaluated at every query point.

Four choices go into this form.

- *The function class* $cal(F)$ specifies the functions that may be fitted.
- *The distance rule* specifies how differences between input delay vectors enter the loss weights or the kernel.
- *The loss weights* $w_n(q)$ specify the contribution of each squared residual.
- *The penalty* $lambda Omega(f)$ controls complexity and stabilizes underdetermined fits.

Simplex projection takes $cal(F)$ to be the constant maps, truncates the weights to the $k$ nearest neighbours, and needs no penalty.
S-map takes $cal(F)$ to be the affine maps, gives weight to every library point, and carries a Tikhonov penalty.
GP-EDM takes $cal(F)$ to be a reproducing kernel Hilbert space, weights every library point equally, and carries the squared norm of that space as its penalty.
Simplex projection represents $F$ by a constant over a truncated neighbourhood.
S-map represents $F$ by an affine function and changes the effective neighbourhood continuously.
GP-EDM fits one function in a reproducing kernel Hilbert space and lets the kernel describe which inputs should have similar function values.

=== The common linear-smoother form

For the three choices just described, the objective is quadratic in the fitted coefficients.
With positive simplex weights, positive S-map regularization and positive GP noise variance, the corresponding quadratic has a unique minimizer.
After all tuning parameters are fixed, the prediction is linear in the library outputs,

$ hat(F)(q) = sum_(n=1)^N b_n (q) thin Y_n $

where the coefficients $b_n(q)$ depend on $X$, $q$ and the fixed estimator parameters.
Write $b(q) in bb(R)^N$ for the vector they form, so that $hat(F)(q) = b(q)^top Y$.
Stacking $b(q)^top$ over the $M$ query points gives an $M times N$ matrix $B$ with $hat(Y) = B Y$, called the *smoother matrix*; the three methods differ only in its rows.

- *Additional outputs reuse the smoother.* When the parameters are fixed and shared across output coordinates, adding columns to $Y$ applies the same $B$ to each column.
- *Changing library times requires a new smoother.* Removing paired rows from $X$ and $Y$ leaves $cal(J)$, $cal(K)$ and the estimator family unchanged, but the coefficients must be recomputed from the remaining rows.
- *The signs and sum of a row determine extrapolation.* The prediction lies in the convex hull of the $Y_n$ for every $Y$ exactly when the entries of $b(q)$ are nonnegative and sum to one. A row that only sums to one reproduces constant outputs but may extrapolate beyond their convex hull.

One qualification.
$B$ is independent of $Y$ only after the parameters governing $w$, $cal(F)$ and $lambda$ have been fixed.
Selecting $k$, $theta$ or kernel parameters from $Y$ makes the complete fitting procedure nonlinear in $Y$.
Prediction error used for parameter selection is also optimistic for that selected configuration unless evaluation uses observations excluded from selection.

== 6. Simplex projection

Simplex projection is the case in which $cal(F)$ is the constant maps and the weights are cut off beyond the $k$ nearest neighbours.
It treats $F$ as constant across the selected neighbourhood, so its smoothing bias grows when $F$ varies appreciably within that neighbourhood#footnote[Sugihara & May (1990), “Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series,” _Nature_ 344(6268):734--741, introduced nonlinear forecasting with nearest neighbours to ecology.].

Write $delta_n = norm(q - X_n)$ for the Euclidean distance from the query point $q$ to each library point, take the $k$ smallest, and write $n_1, dots, n_k$ for their indices and $delta_((1)) <= dots <= delta_((k))$ for the distances.
The conventional choice $k = E + 1$ equals the number of vertices of an $E$-dimensional simplex.
The selected neighbours need not enclose $q$, so the name describes their number rather than a geometric containment guarantee.

The weights are exponential in the distance, normalized by the distance to the nearest neighbour,

$ w_(n_a) = exp(- delta_((a)) / delta_((1))), quad a = 1, dots, k $

and $w_n = 0$ at every library point outside the $k$ nearest.
This cut-off reduces the sum over all $N$ library points to a sum over $k$ of them.
The displayed formula assumes $delta_((1)) > 0$.
Its limit as $delta_((1))$ approaches zero assigns weight only to neighbours at zero distance; an implementation must state how it handles exact duplicates.

This normalization does two things.
The weights become invariant to the overall scale of the distances, and the bandwidth narrows automatically where neighbours are dense.
The nearest neighbour always takes weight $e^(-1)$, so the distance from the query point to its neighbourhood enters only through the ratios $delta_((a)) \/ delta_((1))$.

Since the function class contains only constant maps, the minimization reduces to a weighted mean and no penalty is needed.
The selected weights have a positive sum, so the minimizer is unique.

$ hat(F)(q) = (sum_(a=1)^k w_(n_a) thin Y_(n_a)) / (sum_(a=1)^k w_(n_a)) $

So $b_(n_a)(q) = w_(n_a) \/ sum_c w_(n_c)$ at the $k$ nearest library points, and $b_n (q) = 0$ at all the others.
They are nonnegative and sum to one, so the prediction lies in the convex hull of the neighbouring output vectors.

== 7. S-map

S-map is the case in which $cal(F)$ is the affine maps and every library point is given a nonzero weight.
An affine fit represents the first-order variation of a smooth regression function within the effective neighbourhood and thereby reduces the leading bias of a local constant fit#footnote[Sugihara (1994), “Nonlinear forecasting for the classification of natural time series,” _Philosophical Transactions of the Royal Society of London A_ 348(1688):477--495.].

Write $overline(delta)(q) = 1/N sum_n delta_n$ for the mean distance seen from the query point $q$ and set

$ w_n = exp(- theta thin delta_n / (overline(delta)(q))) $

At $theta = 0$ every weight is $1$ and the fit reduces to a single linear regression over the whole library.
Raising $theta$ concentrates the weight on the nearest neighbours and makes the fit local.
The parameter $theta$ tunes the bandwidth continuously, in place of the discrete cut-off at $k$.

The function class is the affine maps $f(p) = c_0 + C_1^top p$.
To absorb the intercept, extend the design matrix to $tilde(X) = [bold(1), X] in bb(R)^(N times (E+1))$ and the query point to $tilde(q) = (1, q) in bb(R)^(E+1)$, so that the coefficient matrix $C in bb(R)^((E+1) times E')$ carries $c_0^top$ on its first row and $C_1$ on the remaining $E$.

Redundant delay coordinates can make the columns of $tilde(X)$ linearly dependent or nearly dependent.
A large $theta$ can create the same numerical problem by leaving only a few library points with appreciable weight.
Autocorrelation often makes lagged coordinates similar, but it does not make them exactly collinear by definition.

The penalty is taken to be Tikhonov on the linear part, $Omega(f) = norm(C_1)_F^2$, and with $W = op("diag")(w_1, dots, w_N)$ the normal equations read $G C = tilde(X)^top W Y$ with

$ G = tilde(X)^top W tilde(X) + alpha thin op("tr")(tilde(X)^top W tilde(X)) thin I_0 $

where $I_0$ is the identity with its $(1,1)$ entry set to $0$, so that the intercept is left unpenalized.
Multiplying the penalty by the trace makes $alpha$ a dimensionless multiplier of the local normal matrix.
It does not make the fit invariant to rescaling one input coordinate relative to another.
The solution gives the prediction $hat(F)(q) = tilde(q)^top C$, and $b(q)^top = tilde(q)^top G^(-1) tilde(X)^top W$.

$G$ is positive definite whenever $alpha > 0$ and some weight is nonzero, since

$ v^top G v = norm(W^(1\/2) tilde(X) v)^2 + alpha thin op("tr")(tilde(X)^top W tilde(X)) thin v^top I_0 v $

and the two vanish together only at $v = 0$: the second term forces $v_2 = dots = v_(E+1) = 0$, and the first then reads $v_1^2 sum_n w_n$.

Every library point has a nonzero weight, so no entry of that row is structurally zero, and the entries can be negative: the prediction is not confined to the convex hull of $Y$.
Leaving the intercept unpenalized does keep the row summing to one, since a constant $Y$ is then fitted with zero residual and zero penalty, so a constant library is still reproduced exactly.
The weights depend on the query point and so does $C$, so one system is solved per query point.

The fitted slopes $C_1^top$ are coefficients of the locally weighted affine approximation.
They estimate the Jacobian $F'(q)$ only in an asymptotic regime where the effective neighbourhood shrinks, the local sample size grows, and $F$ has the required derivatives#footnote[Masry (1996), “Multivariate local polynomial regression for time series: uniform strong consistency and rates,” _Journal of Time Series Analysis_ 17(6):571--599, proves derivative consistency under stationarity, smoothness, bandwidth and strong-mixing conditions.].
When the input delay-coordinate map embeds $A$ in the deterministic model, this Jacobian is the derivative of $Psi_cal(K) compose Psi_cal(J)^(-1)$.
With non-injective, noisy or stochastic input coordinates, it is the derivative of the conditional mean.
A finite-sample S-map coefficient is therefore an estimate of a local predictive slope, not automatically a structural interaction coefficient.

An improvement in out-of-sample prediction at $theta > 0$ shows that state-dependent weighting predicts better than the global affine fit on that evaluation distribution.
This comparison supports state dependence of the predictive relationship only after alternatives such as nonstationarity, leakage and parameter-selection bias have been controlled.
The observations used to select $theta$ cannot also provide an unbiased estimate of the selected model's improvement.

== 8. GP-EDM

GP-EDM, empirical dynamic modelling with a Gaussian process, uses equal loss weights and a reproducing kernel Hilbert space as its function class#footnote[Munch, Poynor & Arriaza (2017), “Circumventing structural uncertainty: a Bayesian perspective on nonlinear forecasting for ecology,” _Ecological Complexity_ 32:134--143. Munch, Rogers & Sugihara (2023), “Recent developments in empirical dynamic modelling,” _Methods in Ecology and Evolution_ 14(3):732--745. Rasmussen & Williams (2006), _Gaussian Processes for Machine Learning_, MIT Press, chapter 2, gives the Gaussian-process identities used here.].
The kernel encodes how rapidly function values may change with separation in each input coordinate.
Its scale parameters are estimated from data or assigned prior distributions rather than fixed by a local polynomial degree.

Fix a positive definite kernel $kappa$ on $bb(R)^E$, write $cal(H)_kappa$ for its reproducing kernel Hilbert space, and take

$ w_n equiv 1, quad cal(F) = cal(H)_kappa, quad Omega(f) = norm(f)^2_(cal(H)_kappa), quad lambda = sigma^2 $

For one output coordinate, the minimization has a closed solution.
Write $K in bb(R)^(N times N)$ for the matrix with entries $kappa(X_m, X_n)$ and $bold(kappa)(q) in bb(R)^N$ for the vector with entries $kappa(q, X_n)$.

$ hat(F)(q) = bold(kappa)(q)^top (K + sigma^2 I)^(-1) Y $

Here $b(q)^top = bold(kappa)(q)^top (K + sigma^2 I)^(-1)$.
The inverse exists because $K$ is positive semidefinite and $sigma^2 > 0$, which makes $K + sigma^2 I$ positive definite.
The equal weights are query-independent, so one function is fitted and then evaluated at all $M$ query points, rather than the $M$ separate fits the other two methods require.
The prediction is still linear in $Y$ with query-dependent coefficients, and the inverse is computed once and reused across the whole query.

The same expression is the posterior mean of a Gaussian process prior $f tilde cal(G)cal(P)(0, kappa)$ observed with independent Gaussian noise of variance $sigma^2$.
The model also gives the posterior predictive variance for a new noisy output,

$ hat(sigma)^2 (q) = kappa(q, q) - bold(kappa)(q)^top (K + sigma^2 I)^(-1) bold(kappa)(q) + sigma^2 $

This quantity is calibrated as conditional uncertainty only when the Gaussian process covariance, mean and noise model are adequate.
It is model-based uncertainty, whereas the regression function $F$ is defined without a Gaussian process assumption.

=== Kernel similarity

Equal loss weights make GP-EDM a single global fit.
Local similarity enters through the squared-exponential kernel, with one inverse squared length-scale parameter per input coordinate,

$ kappa(p, p') = eta thin exp(- sum_(j=1)^E phi_j (p_j - p'_j)^2), quad phi_j >= 0 $

The direct covariance $kappa(q, X_n)$ decreases as the scaled separation from $q$ increases.
The final smoother coefficient $b_n(q)$ also contains $(K + sigma^2 I)^(-1)$, so it need not decrease monotonically with distance and can be negative.
The $phi_j$ play the bandwidth role inside the function class that $k$ and $theta$ play inside the loss weights.

Two things change with the move.

The kernel scale is not renormalized separately at each query point.
Simplex projection divides distances by the nearest-neighbour distance, while S-map divides them by the mean distance from the current query.
A stationary GP kernel uses the same $phi_j$ at every query point.

The decay is anisotropic.
A separate $phi_j$ per coordinate defines a fitted diagonal squared-distance rule in place of an unscaled Euclidean distance.
This parameterization is called *automatic relevance determination* (ARD).
When $phi_j$ is near zero, changing coordinate $j$ has little effect on the fitted covariance and the coordinate has little predictive relevance for that output.
ARD does not show that the remaining coordinates embed the state; predictive relevance for a specified output and state reconstruction are different properties.

Rows of $B$ are generally dense and signed, as in S-map, and they are not constrained to sum to one.
The $sigma^2$ in the inverse shrinks the prediction toward the prior mean, and by more where the library is sparse.
Consequently, exact reproduction of a constant $Y$ is not guaranteed, and the prediction is not confined to the convex hull of the library outputs.

=== Fitting the kernel

$eta$, the $phi_j$ and $sigma^2$ can be chosen by maximizing the marginal likelihood of the library.
For one output coordinate and a zero prior mean, the log marginal likelihood is

$
  log p(Y | X) = - 1/2 Y^top (K + sigma^2 I)^(-1) Y
  - 1/2 log det (K + sigma^2 I) - N/2 log 2 pi
$

The quadratic term measures agreement with the observed output under the covariance model.
The log-determinant term accounts for the volume of output vectors to which that covariance assigns substantial density.
Together they balance data agreement against covariance flexibility.
A prior on the hyperparameters changes the criterion from maximum likelihood to a posterior mode; the GP-EDM formulation of Munch, Poynor and Arriaza uses an ARD prior that favours negligible effects unless the data support them.
With $E' > 1$ this is one Gaussian process per column of $Y$.
Sharing the hyperparameters across the columns keeps a single smoother matrix, and with it the freedom to widen $Y$ at no cost; fitting them per column gives one smoother matrix per column instead, and that freedom is lost.

The effect on input-coordinate relevance is the reason the fitting step matters here.
One may supply a broad set of candidate variables and lags in $cal(J)$ and use the fitted $phi_j$ to identify coordinates that affect prediction of the chosen output.
This procedure selects predictors for that regression function.
Injectivity remains a separate property governed by the delay-coordinate map.

Hyperparameter fitting changes the linear-smoother property.
Once $eta$, $phi$ and $sigma^2$ are fitted from $Y$, the smoother matrix depends on $Y$, and the complete fitting procedure is not linear in $Y$.
The same dependence arises when $k$ or $theta$ is selected by predictive performance.
A higher-dimensional hyperparameter search increases the need for regularization and evaluation on observations not used for selection.

== 9. The three methods compared

The table collects the statistical choices made by each method.

#block(breakable: false)[
  #table(
    columns: 4,
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([], [*Simplex projection*], [*S-map*], [*GP-EDM*]),
    [Assumed shape of $F$ near $q$], [constant], [affine], [kernel-controlled smoothness],
    [Function class $cal(F)$], [constant maps], [affine maps], [$cal(H)_kappa$],
    [Input-coordinate distance], [Euclidean], [Euclidean], [diagonal, fitted ($phi$)],
    [Weights $w_n (q)$],
    [$exp(-delta_((a)) \/ delta_((1)))$],
    [$exp(-theta thin delta_n \/ overline(delta)(q))$],
    [$1$],

    [Support of the weights], [the $k$ nearest], [all $N$], [all $N$],
    [Bandwidth],
    [per query, from $delta_((1))$],
    [per query, from $overline(delta)(q)$],
    [fixed across query points, from $phi$],

    [Penalty], [none], [Tikhonov on $C_1$, strength $alpha$], [$sigma^2 norm(f)^2_(cal(H)_kappa)$],
    [Row of $B$],
    [$k$ nonzero, nonnegative, sums to $1$],
    [generally dense, signed, sums to $1$],
    [generally dense, signed, sum unconstrained],

    [Prediction], [in the convex hull of $Y$], [unrestricted], [shrunk toward the prior mean],
    [Uncertainty from the fitted model], [not supplied], [not supplied], [predictive variance],
    [Solved per query],
    [$k$ neighbours and $k$ weights],
    [an $(E+1) times (E+1)$ system],
    [nothing; one $N times N$ solve serves all],
  )
]

== 10. Designing input and output coordinates

An analysis specifies $cal(J)$ for the input and $cal(K)$ for the output.
$X$ and $Q$ use the same $cal(J)$ because their rows are compared in one input-coordinate space.
$Y$ uses $cal(K)$, whose variables, lags and dimension may differ from those of $cal(J)$.
The same pair of delay coordinates can be used with any of the three estimators.

The examples take two variables ($d = 2$).
An output horizon of $h$ is represented by an output lag of $-h$.

#block(breakable: false)[
  #table(
    columns: 4,
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Goal*], [*Input $cal(J)$ ($X$, $Q$)*], [*Output $cal(K)$ ($Y$)*], [*Sizes ($X$, $Y$)*]),
    [One variable, horizon $0$], [$((1, 0), (1, tau))$], [$((1, 0))$], [$N times 2$, $N times 1$],

    [One variable to another, $h$ ahead], [$((1, 0), (1, tau))$], [$((2, -h))$], [$N times 2$, $N times 1$],

    [Several variables to one, $h$ ahead],
    [$((1, 0), (1, tau), (2, 0), (2, tau))$],
    [$((1, -h))$],
    [$N times 4$, $N times 1$],

    [Several variables to several, $h$ ahead],
    [$((1, 0), (1, tau), (2, 0), (2, tau))$],
    [$((1, -h), (2, -h))$],
    [$N times 4$, $N times 2$],

    [Different lags per variable], [$((1, 0), (1, tau), (2, 2 tau))$], [$((1, -h))$], [$N times 3$, $N times 1$],

    [Several horizons], [$((1, 0), (1, tau))$], [$((1, -h_1), (1, -h_2))$], [$N times 2$, $N times 2$],
  )
]

*Same variable at horizon $0$.*
This output is degenerate because $Y$ is the first column of $X$ and is already contained in the input vector.

*One variable to another.*
It asks whether the delay vectors built from variable 1 recover variable 2 at $h$ steps ahead, so a forecast in time and a map between variables are carried out in a single operation.
With $h = 0$ the row is *cross mapping*: the delay vectors built from one variable are asked for the contemporaneous value of another.
*Convergent cross mapping* (CCM) repeats this estimate over increasing library sizes and evaluates how cross-map skill changes with $|L|$.
Under the deterministic coupled-system assumptions of CCM, an effect variable contains information about the state of its causes.
Consequently, using variable 1 delay coordinates to recover variable 2 tests the direction variable 2 $arrow.r$ variable 1, not the reverse.
If variable 1 reconstructs the shared state, increasing the library supplies closer analogues and the cross-map estimate of variable 2 improves before approaching its finite-noise limit.
Convergence is therefore evidence consistent with that causal direction only when common forcing, synchrony, temporal trends, sampling effects and alternative directions have been addressed#footnote[Sugihara, May, Ye, Hsieh, Deyle, Fogarty & Munch (2012), “Detecting causality in complex ecosystems,” _Science_ 338(6106):496--500.].

*Adding input coordinates.*
The number of library points is unchanged while the input-coordinate dimension grows from $2$ to $4$.
The new coordinates may separate states that previously looked identical, but they also make close neighbours harder to find in a finite library unless the added coordinates are redundant on the sampled state set.

*Adding output coordinates.*
The input distances and neighbours are unchanged, and only the number of columns of $Y$ grows.
Several variables can therefore reuse one fixed smoother matrix when estimator parameters are shared across output coordinates.

*Different lags per variable.*
Since $cal(J)$ is only a sequence of (variable, lag) pairs, neither the number of coordinates nor the spacing of the lags has to agree across variables.
The displayed delay coordinates read variable 1 at lags $0$ and $tau$ and variable 2 at lag $2 tau$ only.
This specifies delay coordinates that mix variables and time scales.
Generalized embedding theorems cover such delay coordinates when their regularity, dimension and periodic-orbit conditions hold.

*Several output horizons.*
Each coordinate of $cal(K)$ carries its own lag, so the horizons $h_1$ and $h_2$ are estimated at once.
With fixed estimator parameters, neighbours and smoother coefficients are determined by the input alone, so another horizon adds another output column.
Direct estimation does not feed earlier predictions into later inputs, so it avoids recursive propagation of earlier prediction errors.

One combination the table omits takes $cal(K) = cal(J)$ with every lag shifted by $-h$.
The columns of $Y$ then carry the input coordinates advanced by $h$ steps, so the predicted row has the coordinates required for another application of the same fitted map.
This alignment makes iterated multi-step forecasting possible.

Taking the query to be the library itself, $S = L$, and excluding each library point from its own neighbourhood gives leave-one-out prediction.
A *Theiler window* widens the exclusion by removing every library time within $r$ sampling steps of the query time#footnote[Theiler (1986), “Spurious dimension from correlation algorithms applied to limited time-series data,” _Physical Review A_ 34(3):2427--2432, introduced temporal exclusion to prevent serial dependence from creating spurious near neighbours in correlation-dimension estimation.].
In prediction, this window prevents overlapping or strongly dependent delay vectors from serving as nominally independent validation cases.
The required width depends on the lags, forecast horizon, serial dependence and intended deployment setting.

== 11. Assumptions and limits

The regression function remains meaningful beyond deterministic dynamics, but each interpretation requires its own assumptions.

- *Stable relationship*: library and query observations must be governed by the same conditional mean $F$. A regime change can alter $F$ even when input delay vectors remain numerically close.
- *Support*: a local estimate is supported only where the library contains input delay vectors near the query. A query outside the sampled support requires extrapolation, regardless of the estimator's algebraic ability to return a number.
- *Dynamical interpretation*: interpreting $F$ as $Psi_cal(K) compose Psi_cal(J)^(-1)$ requires deterministic dynamics and an input delay-coordinate map that embeds $A$. Otherwise $F$ is a conditional mean, and causal or structural language needs additional assumptions.
- *Temporal dependence*: ergodicity justifies long-run averages but supplies neither an effective sample size nor a convergence rate. Rates for local regression on time series require stronger dependence conditions such as mixing. Validation splits and uncertainty calculations must preserve the temporal information available at the intended prediction time.
- *Leakage*: output times used for evaluation must be absent from fitting and parameter selection. Overlapping delay windows and temporally adjacent states can also make nominal train and test cases nearly identical. A Theiler window enforces separation when that separation matches the intended prediction task.
- *Coordinate scale*: Euclidean distance changes when one input coordinate is rescaled. Standardization, a physically chosen metric, or fitted kernel scales are modelling choices. ARD estimates relative predictive scales conditional on the units, kernel and hyperprior; it does not remove the need to state those choices.
- *Parameter selection*: choosing $cal(J)$, $cal(K)$, $k$, $theta$, $alpha$ or kernel parameters from prediction performance uses output information. Performance of the selected configuration requires evaluation data that played no part in that selection.
- *Dimension*: finite libraries become sparse as the effective dimension of the input distribution grows. For independent observations on a smooth lower-dimensional manifold, local polynomial regression can attain rates governed by the manifold dimension rather than the ambient coordinate dimension#footnote[Bickel & Li (2007), “Local polynomial regression on unknown manifolds,” _IMS Lecture Notes--Monograph Series_ 54:177--186.]. This theorem does not by itself establish an EDM rate on a fractal attractor with dependent observations. Measurement noise can also spread observations away from a lower-dimensional set, making the ambient dimension relevant in finite samples.
- *Skill measure*: Pearson correlation $rho$ is unchanged by a positive affine transformation of the predictions. It therefore does not detect additive bias or multiplicative miscalibration. Error and calibration measures are needed when their magnitudes matter.

The framework fixes the regression function, the three data matrices and the estimator families.
An analysis additionally fixes the input coordinates, output coordinates, library times, query times, distance scale, tuning procedure and evaluation measure.
