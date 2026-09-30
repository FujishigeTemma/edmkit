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

#set heading(numbering: "1.")

#show heading.where(level: 1): it => {
  counter(math.equation).update(0)
  it
}

#set math.equation(
  numbering: n => {
    numbering("(1.1)", counter(heading).get().first(), n)
  },
  supplement: [Eq.],
)

#outline()
#pagebreak()

= The dynamics and the observations

We index equally spaced times by $t in bb(Z)$ and assume a deterministic dynamical system with a state $z_t$ that evolves according to a time-independent dynamics $Phi$,

$ z_(t+1) = Phi(z_t) $ <dynamics>

Let $A$ be a compact set that is invariant under $Phi$, so $Phi(A) = A$, and that contains the initial state $z_0$.
We assume that $Phi$ is a diffeomorphism on an open neighbourhood of $A$.
Together, invariance of $A$ and invertibility of $Phi$ ensure that $z_t$ lies in $A$ at every $t$.

The state $z_t$ usually cannot be obtained directly.
Instead, at each time $t$, we obtain $d$ observations.
Without measurement noise, these are given by *observation functions* $g_1, dots, g_d$, smooth on that neighbourhood,

$ x_t = (g_1(z_t), dots, g_d(z_t)) in bb(R)^d. $ <observation>

We write $x_t^((i))$ for component $i in {1, dots, d}$ of $x_t$.
For states, observations and delay vectors, uppercase letters denote random variables and lowercase letters denote their values.
Thus a random initial state $Z_0$ in $A$ generates $Z_(t+1) = Phi(Z_t)$ and $X_t = (g_1(Z_t), dots, g_d(Z_t))$.
We observe $x = (x_0, dots, x_(T-1))$ along one trajectory.

= Representing states by delay coordinates

Observing different states through $g_i$ can give the same value.
To distinguish more states, we need additional observations.
In particular, observations at other variables and times can distinguish states that $g_i$ alone cannot.
We therefore combine observations at selected variables and lags.

We specify this selection by an ordered sequence of pairs $(i_j, ell_j)$, each selecting an observation function $g_(i_j)$ and a lag $ell_j$.
We call these the *delay coordinates*.

$
  cal(J) = ( (i_1, ell_1), dots, (i_E, ell_E) ),
  quad i_j in {1, dots, d}, quad ell_j in bb(Z)
$

The $j$th delay coordinate is $(i_j, ell_j)$.
The number $E = |cal(J)|$ is the dimension of the resulting delay vector.
Combining the selected observations gives the *delay vector*

$ U_cal(J)(t) = (X^((i_1))_(t - ell_1), dots, X^((i_E))_(t - ell_E)) in bb(R)^E, quad t in bb(Z) $

Replacing $X_t$ with $x_t$ gives $u_cal(J)(t)$.
It can be computed from the available observations exactly when

$ max_j ell_j <= t <= T-1 + min_j ell_j. $

As a special case, taking one variable at evenly spaced lags gives the classical univariate delay coordinates,

$ cal(J) = ( (1, 0), (1, tau), dots, (1, (E-1) tau) ) $

with lag $tau$.
Allowing $i_j$ to vary gives multivariate delay coordinates.
A negative lag $ell_j$ selects an observation taken $abs(ell_j)$ steps after time $t$.

Although $u_cal(J)(t)$ contains observations at times $t - ell_j$, every entry of $u_cal(J)(t)$ can be determined by the state at the single time $t$.
Indeed, @observation gives

$ x^((i_j))_(t - ell_j) = g_(i_j) (z_(t - ell_j)) $

and invertibility ($Phi$ is a diffeomorphism) makes the forward or inverse iterate of $Phi$ well defined,

$ z_(t - ell_j) = Phi^(-ell_j) (z_t) $

And thus, the observations at times $t - ell_j$ can be expressed in terms of $z_t$,

$ x^((i_j))_(t - ell_j) = g_(i_j) ( Phi^(-ell_j) (z_t) ) $

We define the *delay-coordinate map* $Psi_(cal(J))$ to map each state $z$ to its delay vector by

$ Psi_cal(J) (z) = ( g_(i_1)(Phi^(-ell_1)(z)), dots, g_(i_E)(Phi^(-ell_E)(z)) ) $

By construction,

$ U_cal(J)(t) = Psi_cal(J)(Z_t), quad u_cal(J)(t) = Psi_cal(J)(z_t). $

= Learning about the dynamics through delay coordinates

Although $Phi$ is unknown, we can compute $u_cal(J)(t)$ from $x$ and use relations between delay vectors to learn about the dynamics.

Take two sequences of delay coordinates, $cal(J)$ of dimension $E$ and $cal(K)$ of dimension $E'$, with delay-coordinate maps

$
  Psi_cal(J): A arrow.r bb(R)^E,
  quad Psi_cal(K): A arrow.r bb(R)^(E')
$


We define a function $Gamma: Psi_cal(J)(A) arrow.r bb(R)^(E')$ to express the relation from $Psi_cal(J)(z)$ to $Psi_cal(K)(z)$,

$ Gamma(Psi_cal(J)(z)) = Psi_cal(K)(z), quad z in A $ <consistency>

We require $Gamma$ to be single-valued because we want to consider the case where $Psi_cal(J)(z)$ contains sufficient information to determine $Psi_cal(K)(z)$.
That is, we require

$ Psi_cal(J)(z) = Psi_cal(J)(z') quad => quad Psi_cal(K)(z) = Psi_cal(K)(z'), quad z, z' in A $ <fiber>

If $cal(J)$ and $cal(K)$ satisfy @fiber, then there exists a unique function $Gamma$ satisfying @consistency.

Still, we cannot determine $Gamma$ because the maps $Psi_cal(J)$ and $Psi_cal(K)$ depend on the unknown $Phi$.
We therefore estimate $Gamma$ from $(u_cal(J)(t), u_cal(K)(t))$.

Let $hat(Gamma)$ denote an estimated function on the same domain as $Gamma$.
$Gamma$ is fixed by the underlying system and chosen coordinates, whereas $hat(Gamma)$ depends on $x$ and the estimation method.
Our aim is

$ hat(Gamma)(q) = Gamma(q), quad forall q in Psi_cal(J)(A). $ <recovery>

Because $x$ is finite, the observed pairs $(u_cal(J)(t), u_cal(K)(t))$ are available at only finitely many times $t$.
By @consistency, each pair gives one exact value of $Gamma$,

$ Gamma(u_cal(J)(t)) = u_cal(K)(t) $.

Thus @recovery requires

$ hat(Gamma)(u_cal(J)(t)) = u_cal(K)(t) $ <agreement-on-observed-pairs>

for every observed pair.
This condition alone does not imply @recovery because multiple functions can agree with all these pairs while taking different values elsewhere.

To estimate $Gamma(q)$ at such points, we use the continuity of $Gamma$ on $Psi_cal(J)(A)$.
This continuity follows from our assumptions, as shown below.
Continuity means that when $u_cal(J)(t)$ is sufficiently close to $q$, the corresponding $u_cal(K)(t) = Gamma(u_cal(J)(t))$ is close to $Gamma(q)$.

Using this property, we can prove @recovery if the observed delay vectors are dense in the domain and $hat(Gamma)$ is continuous and agrees with every observed pair.
Suppose we extend $x$ along the trajectory as $T -> infinity$, and the $u_cal(J)(t)$ for which both delay vectors are computable are dense in $Psi_cal(J)(A)$.
That is, every point in this domain has such delay vectors arbitrarily close to it.
Suppose also that $hat(Gamma)$ is continuous and satisfies @agreement-on-observed-pairs at every time $t$ for which both delay vectors are computable.

To prove @recovery, take any $q in Psi_cal(J)(A)$.
By density, there are times $t_n$ such that $u_cal(J)(t_n) -> q$, with both delay vectors computable at each $t_n$.
By @consistency and the assumed agreement,
$ hat(Gamma)(u_cal(J)(t_n)) = u_cal(K)(t_n) = Gamma(u_cal(J)(t_n)) $
for every $n$.
Since both functions are continuous,
$
  hat(Gamma)(q)
  = lim_(n -> infinity) hat(Gamma)(u_cal(J)(t_n))
  = lim_(n -> infinity) Gamma(u_cal(J)(t_n))
  = Gamma(q).
$

Since $q$ was arbitrary, this proves @recovery.

In practice, the $u_cal(J)(t)$ computable together with $u_cal(K)(t)$ from $x$ need not be dense in $Psi_cal(J)(A)$, even as $T -> infinity$ along the trajectory.
Moreover, the estimation method need not produce a continuous $hat(Gamma)$ or satisfy @agreement-on-observed-pairs exactly.
Without these guarantees, the preceding argument does not establish @recovery.
To assess the estimation error, we can instead bound it using an observed pair, without assuming density, continuity of the estimate, or exact agreement.
Take any $q in Psi_cal(J)(A)$ and any time $t$ for which both delay vectors are computable from $x$.
The triangle inequality gives

$
  norm(hat(Gamma)(q) - Gamma(q))
  <= & norm(hat(Gamma)(q) - hat(Gamma)(u_cal(J)(t))) \
   + & norm(hat(Gamma)(u_cal(J)(t)) - u_cal(K)(t)) \
   + & norm(u_cal(K)(t) - Gamma(q))
$

- The first term measures the change in $hat(Gamma)$ between $u_cal(J)(t)$ and $q$.
- The second is its error at $u_cal(J)(t)$, since @consistency gives $Gamma(u_cal(J)(t)) = u_cal(K)(t)$.
- The third measures the change in $Gamma$ between $u_cal(J)(t)$ and $q$ by the same equality.
Thus the error at $q$ is small if the error at $u_cal(J)(t)$ is small and both functions change little between the two points.

To make the first and third terms small, we need an observed input close to $q$ and little variation in either function between those inputs.
Continuity of $Gamma$ ensures this for the third term, and continuity of $hat(Gamma)$ would do the same for the first, for a fixed estimate.
Together with a small error at the observed input, these properties make the displayed bound small.
As $T$ grows, however, the estimate itself can change.
Convergence therefore requires control of that changing estimate's variation as well as the distances to observed inputs and the errors there.
Continuity of each estimate alone does not supply this control.

To justify the use of continuity in the recovery argument and the third error term, we now show that

$
  q_n -> q quad => quad Gamma(q_n) -> Gamma(q),
  quad q_n, q in Psi_cal(J)(A).
$ <gamma-continuity>

Each coordinate of a delay-coordinate map is a composition $g_i compose Phi^(-ell)$ of smooth functions, so it is continuous.
A finite tuple of continuous functions is continuous, so both $Psi_cal(J)$ and $Psi_cal(K)$ are continuous.

To prove @gamma-continuity, suppose that $q_n -> q$ but $Gamma(q_n)$ does not converge to $Gamma(q)$.
Then some $epsilon > 0$ and a subsequence $q_(n_k)$ satisfy

$ norm(Gamma(q_(n_k)) - Gamma(q)) >= epsilon quad "for every" k. $

For each $k$, choose a state $z^((k)) in A$ with $Psi_cal(J)(z^((k))) = q_(n_k)$.
Here the superscript indexes the chosen states, independently of their times along any trajectory.
Compactness of $A$ gives a further subsequence $z^((k_j)) -> z in A$.
Continuity of $Psi_cal(J)$ gives

$ Psi_cal(J)(z) = lim_(j -> infinity) Psi_cal(J)(z^((k_j))) = q. $

By @consistency and continuity of $Psi_cal(K)$,

$
  Gamma(q_(n_(k_j)))
  = Psi_cal(K)(z^((k_j)))
  -> Psi_cal(K)(z)
  = Gamma(q).
$

This contradicts the lower bound $epsilon$, and proves @gamma-continuity.
Thus the continuity used above follows from the assumptions on the system and @fiber.

The preceding discussion assumes @fiber throughout $A$, which ensures that $Gamma$ exists in the first place.
Values of the delay-coordinate maps at finitely many states on one trajectory cannot establish this condition throughout $A$.
To obtain a sufficient condition on the input map alone, we ask whether its delay vector determines the state.
If it does, that state determines the output for every choice of $cal(K)$.

We call $Psi_cal(J)$ *injective* on $A$ when

$ Psi_cal(J)(z) = Psi_cal(J)(z') quad => quad z = z', quad z, z' in A. $

Under this assumption, we want to establish

$
  Psi_cal(J)(z) = Psi_cal(J)(z') quad => quad Psi_cal(K)(z) = Psi_cal(K)(z')
$

for all $z, z' in A$ and every choice of $cal(K)$.
Indeed, equality of the input vectors implies $z = z'$ by injectivity, and evaluating $Psi_cal(K)$ at the same state gives equal output vectors.
This is @fiber, so injectivity ensures the existence of $Gamma$ for every output coordinate sequence.

To express this relation through the recovered state, we want to write

$ Gamma(q) = (Psi_cal(K) compose Psi_cal(J)^(-1))(q), quad q in Psi_cal(J)(A). $ <gamma>

Injectivity makes $Psi_cal(J)$ a bijection from $A$ to its image, so its inverse is defined there and satisfies

$ Psi_cal(J)^(-1)(Psi_cal(J)(z)) = z, quad z in A. $

For any $q$ in the image, substituting $z = Psi_cal(J)^(-1)(q)$ into @consistency proves @gamma.

To use nearby delay vectors to recover nearby states, we also need continuity of this inverse:

$
  q_n -> q quad => quad Psi_cal(J)^(-1)(q_n) -> Psi_cal(J)^(-1)(q),
  quad q_n, q in Psi_cal(J)(A).
$ <inverse-continuity>

Suppose the conclusion fails, and write $z^((n)) = Psi_cal(J)^(-1)(q_n)$ and $z = Psi_cal(J)^(-1)(q)$.
There is then an open neighbourhood of $z$ outside which a subsequence of $z^((n))$ remains.
By compactness, a further subsequence converges to some $z'$ outside that neighbourhood.
Continuity of $Psi_cal(J)$ gives $Psi_cal(J)(z') = q = Psi_cal(J)(z)$.
Injectivity forces $z' = z$, a contradiction.
This proves @inverse-continuity.
Thus $Psi_cal(J)$ is an *embedding* of $A$ into $bb(R)^E$: it is injective and continuous, and its inverse on its image is continuous.

Like @fiber, injectivity is a property of the maps defined by the unknown $Phi$ and $g_i$ and cannot in general be verified from $(u_cal(J)(t), u_cal(K)(t))$ at finitely many times.
We therefore turn to theoretical conditions on the dynamics and the selected coordinates that imply injectivity.
Embedding theorems provide sufficient conditions under which this conclusion holds for almost every collection of observation functions#footnote[Takens (1981), “Detecting strange attractors in turbulence,” _Dynamical Systems and Turbulence, Warwick 1980_, Lecture Notes in Mathematics 898:366--381, treats consecutive observations on a compact manifold. Sauer, Yorke & Casdagli (1991), “Embedology,” _Journal of Statistical Physics_ 65(3--4):579--616, treats compact invariant sets using box-counting dimension and prevalence. Deyle & Sugihara (2011), “Generalized theorems for nonlinear state space reconstruction,” _PLoS ONE_ 6(3):e18295, theorem 7, treats several observation functions and non-consecutive lags.].
For the multivariate and non-consecutive lags used here, the generalized theorem combines a requirement that $E$ exceed twice the box-counting dimension of $A$ with conditions on periodic points and derivatives of iterates of $Phi$.
These are additional assumptions beyond smoothness and invertibility.
The precise periodic-point conditions depend on the selected lags, so applying the theorem requires checking its full hypotheses for those coordinates.

The conclusion also allows exceptional observation functions for which injectivity may fail.
Thus the theorem supplies a theoretical justification under its hypotheses, but does not verify injectivity for the particular unknown observation functions that produced $x$.

= From $Gamma$ to the regression function

So far, we have assumed deterministic dynamics, exact observations and @fiber.
Under these assumptions, $U_cal(J)(t)$ determines $U_cal(K)(t)$ through $Gamma$,

$ U_cal(K)(t) = Gamma(U_cal(J)(t)) $

at every time $t$.
If @fiber fails, however, we can have $Psi_cal(J)(z) = Psi_cal(J)(z')$ but $Psi_cal(K)(z) != Psi_cal(K)(z')$.
With measurement noise or stochastic dynamics, $U_cal(J)(t)$ is still defined by the selected observations $X_t$, but need not equal $Psi_cal(J)(Z_t)$.
Even when @fiber holds for the underlying system, measurement noise can make different values of $U_cal(K)(t)$ correspond to the same value of $U_cal(J)(t)$.
Stochastic dynamics can also produce such variation.
In any of these cases, $U_cal(J)(t)$ need not determine $U_cal(K)(t)$ uniquely, so a function satisfying the displayed equality at every time need not exist.

Even when the output is not uniquely determined, we want to assign one prediction to each input and use the same function at every time.
The joint distribution of the input and output determines their conditional prediction errors.
We therefore assume that the joint distribution of $(U_cal(J)(t), U_cal(K)(t))$ does not depend on $t$.
We measure prediction error by squared Euclidean distance, so our aim is to find a function $F$ such that

$
  F(q) = op("arg min", limits: #true)_(a in bb(R)^(E'))
  bb(E)[thin norm(U_cal(K)(t) - a)^2 | U_cal(J)(t) = q thin].
$ <conditional-prediction>

We assume that $U_cal(K)(t)$ has a finite second moment.
Conditional expectations, and the minimization in @conditional-prediction, are understood up to sets of input probability zero#footnote[When the input is continuously distributed, the event $U_cal(J)(t) = q$ has probability zero. A regular conditional distribution defines the conditional expectations for input-almost every $q$.].
To solve @conditional-prediction, we define the *regression function* by

$ F(q) = bb(E)[thin U_cal(K)(t) | U_cal(J)(t) = q thin] $ <regression>

and show that, for input-almost every $q$ and every $a in bb(R)^(E')$,

$
  & bb(E)[norm(U_cal(K)(t)-a)^2 | U_cal(J)(t)=q] \
  & quad = bb(E)[norm(U_cal(K)(t)-F(q))^2 | U_cal(J)(t)=q]
  + norm(F(q)-a)^2.
$ <conditional-error-decomposition>

Expand $U_cal(K)(t)-a = (U_cal(K)(t)-F(q)) + (F(q)-a)$ inside the squared norm.
The cross term has conditional expectation

$ 2 (F(q)-a)^top bb(E)[U_cal(K)(t)-F(q) | U_cal(J)(t)=q] = 0 $

by @regression.
The two remaining terms give @conditional-error-decomposition.
Its first term is independent of $a$, and its second is nonnegative and vanishes exactly when $a=F(q)$.
Thus $F(q)$ is the unique minimizer in @conditional-prediction at input-almost every $q$#footnote[Györfi, Kohler, Krzyżak & Walk (2002), _A Distribution-Free Theory of Nonparametric Regression_, Springer Series in Statistics, chapter 1.].
Because the joint distribution is the same at every time, $F$ does not depend on $t$.
Because $F$ is defined from this distribution, it does not depend on the observed sample $x$ or its length $T$.

To justify the assumed time invariance under the deterministic model, take a probability measure $mu$ on $A$ that is invariant under $Phi$ and let $Z_0$ have distribution $mu$.
Invariance gives $Z_t$ distribution $mu$ at every time.
Since the delay-vector pair equals $(Psi_cal(J)(Z_t), Psi_cal(K)(Z_t))$, its joint distribution is also the same at every time.
With measurement noise or stochastic dynamics, we assume this time invariance directly.

To estimate $F(q)$ using outputs at nearby inputs, we need the conditional means at those inputs to be close to $F(q)$.
We therefore assume that the conditional mean has a continuous version on the support of the input distribution, and use that version as $F$.
Here the *support* is the set of inputs every neighbourhood of which has positive probability.
A version agrees with the conditional mean outside a set of input probability zero.

To ensure that this choice specifies one function throughout the support, we show that any two continuous versions $F_1$ and $F_2$ satisfy

$ F_1(q) = F_2(q), quad q "in the input support". $ <continuous-version-uniqueness>

If they differ at a point of the support, continuity makes $norm(F_1-F_2)$ positive throughout some neighbourhood of that point within the support.
That neighbourhood has positive input probability by the definition of support.
This contradicts the fact that both versions equal the same conditional mean almost everywhere, and proves @continuous-version-uniqueness.
We take this support as $op("dom") F$.

To relate this target to the deterministic recovery problem, we now show that, with exact observations and @fiber,

$ F(q) = Gamma(q), quad q in op("dom") F. $ <regression-gamma>

By @consistency, $U_cal(K)(t) = Gamma(U_cal(J)(t))$.
Taking the conditional expectation given $U_cal(J)(t)$ shows that $Gamma$ is a version of the conditional mean.
The set $Psi_cal(J)(A)$ is compact, hence closed, because $A$ is compact and $Psi_cal(J)$ is continuous.
The input lies in this set with probability one, so its support is contained in it.
By @gamma-continuity, $Gamma$ restricted to that support is continuous.
Uniqueness of the continuous version therefore proves @regression-gamma.
Thus estimating $F$ includes the earlier deterministic target on the input support, while also defining a target when outputs are not uniquely determined.

= Training data and query points

To estimate $F$ from $x$, choose training indices $t_1, dots, t_N$ and query indices $s_1, dots, s_M$ for which the required observations are available:

$
  0 <= t_n - ell <= T-1 quad "for every" (i, ell) "in" cal(J) "or" cal(K), quad n = 1, dots, N,
$
$
  0 <= s_m - ell <= T-1 quad "for every" (i, ell) "in" cal(J), quad m = 1, dots, M.
$

Vectors are column vectors.
Matrices use bold capitals, with subscripts $(n,:)$ and $(:,j)$ selecting a row and a column.
Stacking the selected vectors gives

$
  bold(X) = mat(u_cal(J)(t_1)^top; dots.v; u_cal(J)(t_N)^top) in bb(R)^(N times E),
  quad bold(Y) = mat(u_cal(K)(t_1)^top; dots.v; u_cal(K)(t_N)^top) in bb(R)^(N times E'),
$
$
  bold(Q) = mat(u_cal(J)(s_1)^top; dots.v; u_cal(J)(s_M)^top) in bb(R)^(M times E).
$

For example, let $d = 1$, $x_t = t$ for $t = 0, 1, dots, 9$, and $cal(J) = ((1, 0), (1, 2), (1, 4))$.
Then $u_cal(J)(t)$ can be computed for $t = 4, 5, dots, 9$, so choosing these as training indices gives

$
  bold(X) = mat(
    4, 2, 0;
    5, 3, 1;
    6, 4, 2;
    7, 5, 3;
    8, 6, 4;
    9, 7, 5
  ).
$

Each row corresponds to one time and each column to one delay coordinate.
$bold(X)$ contains training inputs, $bold(Y)$ contains the corresponding training outputs, and $bold(Q)$ contains query points.

#block(breakable: false)[
  #table(
    columns: 4,
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header([*Matrix*], [*Size*], [*One row per*], [*One column per*]),
    [$bold(X)$], [$N times E$], [training pair], [input coordinate],
    [$bold(Y)$], [$N times E'$], [the same training pair], [output coordinate],
    [$bold(Q)$], [$M times E$], [query point], [input coordinate],
  )
]

The coordinate sequences and indices encode four parts of the estimation problem.

- $cal(J)$ selects the variables and lags used as input coordinates.
- $cal(K)$ selects the variables and lags used as output coordinates, including the forecast horizon.
- $t_1, dots, t_N$ select the training pairs.
- $s_1, dots, s_M$ select the query points.


From here on, we compute the prediction for one $q = bold(Q)_(m,:)^top$ at a time.
The notation applies row by row even when several predictions share one fitted function or one set of tuning parameters.

= Estimating $F$ from training pairs

We want to estimate $F(q)$ using the training pairs, although their inputs need not equal $q$ and their outputs need not equal their conditional means.
The difference between an output and its conditional mean is its *residual*.
By @regression,

$ bb(E)[U_cal(K)(t)-F(U_cal(J)(t)) | U_cal(J)(t)] = 0. $

For the observed pair at $t_n$, the residual is $u_cal(K)(t_n)-F(u_cal(J)(t_n))$.
Continuity of $F$ makes $F(u_cal(J)(t_n))$ close to $F(q)$ when $u_cal(J)(t_n)$ is close to $q$.
To reduce the contribution of the residuals, we also want to average over multiple training pairs.
Because these pairs come from one trajectory, conditional mean zero alone does not guarantee that their realized residuals average to zero.

To connect averages along the trajectory to the distribution defining $F$, we assume in the deterministic model that the invariant measure $mu$ is ergodic under $Phi$.
For each fixed integrable function of the delay-vector pair, ergodicity makes its time average over $t=0, dots, T-1$ converge to its expectation as $T -> infinity$, for $mu$-almost every initial state#footnote[Walters (1982), _An Introduction to Ergodic Theory_, Graduate Texts in Mathematics 79, Springer, gives the ergodic theorem used here.].
With measurement noise or stochastic dynamics, we assume the same convergence of averages over consecutive delay-vector pairs.
For fixed finite coordinate sequences, requiring that both vectors be computable removes only finitely many times at either end of the observed interval and does not change this limit.
This convergence concerns averages of a fixed integrable function along the trajectory.
It does not by itself establish convergence for an arbitrary selection of training indices or for neighbourhoods and weights that change with the sample size.
Those require conditions on the sampling and estimation method as well.

To construct an estimate from the available finite sample, we fit a function to the training outputs, using input similarity to control how observations contribute to the prediction at $q$.
Simplex projection, S-map and GP-EDM implement this through a common weighted least-squares problem.

To specify this fit, each method makes four choices.

- The function class $cal(F)$ restricts the functions that may be fitted.
- The distance rule specifies how separation between input delay vectors enters the loss weights or the kernel.
- The nonnegative loss weight $w_n(q)$ sets the contribution of training pair $n$ to the fit at $q$.
- The penalty $lambda Omega(f)$ suppresses functions with large $Omega(f)$ and stabilizes fits that the weighted data do not determine uniquely.

Given these choices, all three methods solve the regularized weighted least-squares problem

$
  hat(f)_q = op("arg min", limits: #true)_(f in cal(F)) thin
  sum_(n=1)^N w_n (q) thin norm(bold(Y)_(n,:)^top - f(bold(X)_(n,:)^top))^2 + lambda thin Omega(f),
  quad hat(F)(q) = hat(f)_q (q)
$

Because the optimization returns a fitted function rather than only a value at $q$, evaluating its minimizer at $q$ gives the estimate $hat(F)(q)$.

Simplex projection restricts $cal(F)$ to constant maps, assigns nonzero weights only to the $k$ nearest inputs and sets the penalty to zero.
S-map restricts $cal(F)$ to affine maps, assigns every training pair a query-dependent weight and uses a Tikhonov penalty.
GP-EDM takes $cal(F)$ to be a reproducing kernel Hilbert space, assigns every training pair the same loss weight and penalizes the squared norm in that space.
Thus simplex projection and S-map put input similarity in the weights, whereas GP-EDM puts it in the kernel that defines $cal(F)$.
This choice also determines whether fitting must be repeated: query-dependent weights make $hat(f)_q$ a separate fit for each query, whereas equal weights give one fitted function that can be evaluated at every query.

To compare how the fitted predictions use the training outputs, we will express each method's solution in the form

$ hat(F)(q) = sum_(n=1)^N b_n(q) thin bold(Y)_(n,:)^top = bold(Y)^top b(q), $

where $b(q) = (b_1(q), dots, b_N(q))^top$.
The following sections derive these coefficients from each method's minimization problem, with its tuning parameters held fixed.
To express predictions at all query points, let row $m$ of $bold(B) in bb(R)^(M times N)$ be $b(bold(Q)_(m,:)^top)^top$ and row $m$ of $hat(bold(Y)) in bb(R)^(M times E')$ be $hat(F)(bold(Q)_(m,:)^top)^top$.
Then the displayed representation gives

$ hat(bold(Y)) = bold(B) bold(Y). $

This notation lets us compare the methods through their coefficients: which training outputs contribute, whether coefficients can be negative, and whether they sum to one.
The parameters must be held fixed for this comparison of linear dependence on the training outputs.
If they are selected using $bold(Y)$, the resulting matrix can depend on $bold(Y)$ through that selection, and the complete procedure need not be linear in $bold(Y)$.

= Simplex projection

Simplex projection is the case in which $cal(F)$ is the constant maps and the weights are cut off beyond the $k$ nearest neighbours.
It treats $F$ as constant across the selected neighbourhood, so its smoothing bias grows when $F$ varies appreciably within that neighbourhood#footnote[Sugihara & May (1990), “Nonlinear forecasting as a way of distinguishing chaos from measurement error in time series,” _Nature_ 344(6268):734--741, introduced nonlinear forecasting with nearest neighbours to ecology.].

Write $delta_n = norm(q - bold(X)_(n,:)^top)$ for the Euclidean distance from the query point $q$ to each training input, take the $k$ smallest, and write $n_1, dots, n_k$ for their indices and $delta_((1)) <= dots <= delta_((k))$ for the distances.
The conventional choice $k = E + 1$ equals the number of vertices of an $E$-dimensional simplex.
However, the selected neighbours need not enclose $q$, so the name describes their number rather than a geometric containment guarantee.

The weights are exponential in the distance, normalized by the distance to the nearest neighbour,

$ w_(n_a) = exp(- delta_((a)) / delta_((1))), quad a = 1, dots, k $

and $w_n = 0$ at every training input outside the $k$ nearest.
This cut-off reduces the sum over all $N$ training pairs to a sum over $k$ of them.
The displayed formula assumes $delta_((1)) > 0$.
Its limit as $delta_((1))$ approaches zero assigns weight only to neighbours at zero distance; an implementation must state how it handles exact duplicates.

This normalization does two things.
The weights become invariant to the overall scale of the distances, and the bandwidth narrows automatically where neighbours are dense.
The nearest neighbour always takes weight $e^(-1)$, so the distance from the query point to its neighbourhood enters only through the ratios $delta_((a)) \/ delta_((1))$.

Since the function class contains only constant maps, the minimization reduces to a weighted mean and no penalty is needed.
The selected weights have a positive sum, so the minimizer is unique.

$ hat(F)(q) = (sum_(a=1)^k w_(n_a) thin bold(Y)_(n_a,:)^top) / (sum_(a=1)^k w_(n_a)) $

So $b_(n_a)(q) = w_(n_a) \/ sum_c w_(n_c)$ at the $k$ nearest training inputs, and $b_n (q) = 0$ at all the others.
They are nonnegative and sum to one, so the prediction lies in the convex hull of the neighbouring output vectors.

= S-map

S-map is the case in which $cal(F)$ is the affine maps and every training pair is given a nonzero weight.
An affine fit represents the first-order variation of a smooth regression function within the effective neighbourhood and thereby reduces the leading bias of a local constant fit#footnote[Sugihara (1994), “Nonlinear forecasting for the classification of natural time series,” _Philosophical Transactions of the Royal Society of London A_ 348(1688):477--495.].

Write $overline(delta)(q) = 1/N sum_n delta_n$ for the mean distance seen from the query point $q$ and set

$ w_n = exp(- theta thin delta_n / (overline(delta)(q))) $

At $theta = 0$ every weight is $1$ and the fit reduces to a single linear regression over the whole training set.
Raising $theta$ concentrates the weight on the nearest neighbours and makes the fit local.
Thus, $theta$ tunes the bandwidth continuously, in place of the discrete cut-off at $k$.

With these weights specified, we write the affine maps as $f(p) = c_0 + bold(C)_1^top p$.
To absorb the intercept, extend the design matrix to $tilde(bold(X)) = [bold(1), bold(X)] in bb(R)^(N times (E+1))$ and the query point to $tilde(q) = mat(1; q) in bb(R)^(E+1)$, so that the coefficient matrix $bold(C) in bb(R)^((E+1) times E')$ carries $c_0^top$ on its first row and $bold(C)_1$ on the remaining $E$.

Redundant delay coordinates can make the columns of $tilde(bold(X))$ linearly dependent or nearly dependent.
A large $theta$ can create the same numerical problem by leaving only a few training pairs with appreciable weight.
Autocorrelation often makes lagged coordinates similar, but it does not make them exactly collinear by definition.

To stabilize the fit in these cases, we use a Tikhonov penalty on the linear part, $Omega(f) = norm(bold(C)_1)_F^2$.
With $bold(W) = op("diag")(w_1, dots, w_N)$, the resulting normal equations read $bold(G) bold(C) = tilde(bold(X))^top bold(W) bold(Y)$ with

$
  bold(G) = tilde(bold(X))^top bold(W) tilde(bold(X)) + alpha thin op("tr")(tilde(bold(X))^top bold(W) tilde(bold(X))) thin bold(I)_0
$

where $bold(I)_0$ is the identity with its $(1,1)$ entry set to $0$, so that the intercept is left unpenalized.
Multiplying the penalty by the trace makes $alpha$ a dimensionless multiplier of the local normal matrix.
This normalization does not, however, make the fit invariant to rescaling one input coordinate relative to another.
The solution gives the prediction $hat(F)(q) = bold(C)^top tilde(q)$, and $b(q)^top = tilde(q)^top bold(G)^(-1) tilde(bold(X))^top bold(W)$.

$bold(G)$ is positive definite whenever $alpha > 0$ and some weight is nonzero, since

$
  v^top bold(G) v = norm(bold(W)^(1\/2) tilde(bold(X)) v)^2 + alpha thin op("tr")(tilde(bold(X))^top bold(W) tilde(bold(X))) thin v^top bold(I)_0 v
$

and the two vanish together only at $v = 0$: the second term forces $v_2 = dots = v_(E+1) = 0$, and the first then reads $v_1^2 sum_n w_n$.

Every training pair has a nonzero weight, so no entry of that row is structurally zero, and the entries can be negative: the prediction is not confined to the convex hull of the target vectors $bold(Y)_(n,:)^top$.
Leaving the intercept unpenalized does keep the row summing to one, since constant training targets are then fitted with zero residual and zero penalty, so they are still reproduced exactly.
The weights depend on the query point and so does $bold(C)$, so one system is solved per query point.

Besides giving predictions, the affine fit provides slopes $bold(C)_1^top$ that describe local variation in the fitted function.
They estimate the Jacobian $F'(q)$ only in an asymptotic regime where the effective neighbourhood shrinks, the local sample size grows, and $F$ has the required derivatives#footnote[Masry (1996), “Multivariate local polynomial regression for time series: uniform strong consistency and rates,” _Journal of Time Series Analysis_ 17(6):571--599, proves derivative consistency under stationarity, smoothness, bandwidth and strong-mixing conditions.].
When the input delay-coordinate map embeds $A$ in the deterministic model, this Jacobian is the derivative of $Psi_cal(K) compose Psi_cal(J)^(-1)$.
With non-injective, noisy or stochastic input coordinates, it is the derivative of the conditional mean.
A finite-sample S-map coefficient is therefore an estimate of a local predictive slope, not automatically a structural interaction coefficient.

An improvement in out-of-sample prediction at $theta > 0$ shows that state-dependent weighting predicts better than the global affine fit on that evaluation distribution.
This comparison supports state dependence of the predictive relationship only after alternatives such as nonstationarity, leakage and parameter-selection bias have been controlled.
In particular, the pairs used to select $theta$ cannot also provide an unbiased estimate of the selected model's improvement.

= GP-EDM

GP-EDM, empirical dynamic modelling with a Gaussian process, uses equal loss weights and a reproducing kernel Hilbert space as its function class#footnote[Munch, Poynor & Arriaza (2017), “Circumventing structural uncertainty: a Bayesian perspective on nonlinear forecasting for ecology,” _Ecological Complexity_ 32:134--143. Munch, Rogers & Sugihara (2023), “Recent developments in empirical dynamic modelling,” _Methods in Ecology and Evolution_ 14(3):732--745. Rasmussen & Williams (2006), _Gaussian Processes for Machine Learning_, MIT Press, chapter 2, gives the Gaussian-process identities used here.].
The kernel encodes how rapidly function values may change with separation in each input coordinate.
Its scale parameters are estimated from $bold(X)$ and $bold(Y)$ or assigned prior distributions rather than fixed by a local polynomial degree.

Fix a positive definite kernel $kappa$ on $bb(R)^E$, write $cal(H)_kappa$ for its reproducing kernel Hilbert space, and take

$ w_n equiv 1, quad cal(F) = cal(H)_kappa, $
$ Omega(f) = norm(f)^2_(cal(H)_kappa), quad lambda = sigma^2. $

For one output coordinate, the minimization has a closed solution.
To express this solution, write $bold(K) in bb(R)^(N times N)$ with $bold(K)_(n,n') = kappa(bold(X)_(n,:)^top, bold(X)_(n',:)^top)$, and let $kappa_q in bb(R)^N$ have entries $(kappa_q)_n = kappa(q, bold(X)_(n,:)^top)$.

$ hat(F)(q) = bold(Y)^top (bold(K) + sigma^2 bold(I))^(-1) kappa_q $

Here $b(q)^top = kappa_q^top (bold(K) + sigma^2 bold(I))^(-1)$.
The inverse exists because $bold(K)$ is positive semidefinite and $sigma^2 > 0$, which makes $bold(K) + sigma^2 bold(I)$ positive definite.
The equal weights are query-independent, so one function is fitted and then evaluated at all $M$ query points, rather than the $M$ separate fits the other two methods require.
The prediction is still linear in $bold(Y)$ with query-dependent coefficients, and the inverse is computed once and reused across the whole query.

Under a probabilistic interpretation, placing a zero-mean Gaussian process prior with covariance $kappa$ on the candidate functions and assuming independent Gaussian noise of variance $sigma^2$ gives the same expression for the posterior mean.
The model also gives the posterior predictive variance for a new noisy output,

$ hat(sigma)^2 (q) = kappa(q, q) - kappa_q^top (bold(K) + sigma^2 bold(I))^(-1) kappa_q + sigma^2 $

This uncertainty estimate depends on the probabilistic assumptions: it is calibrated as conditional uncertainty only when the Gaussian process covariance, mean and noise model are adequate.
It is model-based uncertainty, whereas the regression function $F$ is defined without a Gaussian process assumption.

== Kernel similarity

Equal loss weights make GP-EDM a single global fit.
Local similarity enters through the squared-exponential kernel, with one inverse squared length-scale parameter per input coordinate,

$ kappa(p, p') = eta thin exp(- sum_(j=1)^E phi_j (p_j - p'_j)^2), quad phi_j >= 0 $

The direct covariance $kappa(q, bold(X)_(n,:)^top)$ decreases as the scaled separation from $q$ increases.
The final smoother coefficient $b_n(q)$, however, also contains $(bold(K) + sigma^2 bold(I))^(-1)$, so it need not decrease monotonically with distance and can be negative.
The $phi_j$ play the bandwidth role inside the function class that $k$ and $theta$ play inside the loss weights.

Using these kernel parameters to determine similarity differs from the distance weights of simplex projection and S-map in two respects.

First, the kernel scale is not renormalized separately at each query point.
Simplex projection divides distances by the nearest-neighbour distance, while S-map divides them by the mean distance from the current query.
In contrast, a stationary GP kernel uses the same $phi_j$ at every query point.

Second, the covariance can decay at different rates along different coordinates.
A separate $phi_j$ per coordinate defines a fitted diagonal squared-distance rule in place of an unscaled Euclidean distance.
This parameterization is called *automatic relevance determination* (ARD).
When $phi_j$ is near zero, changing coordinate $j$ has little effect on the fitted covariance and the coordinate has little predictive relevance for that output.
This predictive relevance does not establish that the remaining coordinates embed the state, because it concerns the specified output rather than whether distinct states have distinct input vectors.

Rows of $bold(B)$ are generally dense and signed, as in S-map, and they are not constrained to sum to one.
The $sigma^2$ in the inverse shrinks the prediction toward the prior mean, and by more where the training inputs are sparse.
Consequently, exact reproduction of constant training targets is not guaranteed, and the prediction is not confined to the convex hull of the training targets.

== Fitting the kernel

$eta$, the $phi_j$ and $sigma^2$ can be chosen by maximizing the marginal likelihood of the training targets given the training inputs.
For output coordinate $j$ and a zero prior mean, the log marginal likelihood is

$
  log p(bold(Y)_(:,j) | bold(X)) = - 1/2 bold(Y)_(:,j)^top (bold(K) + sigma^2 bold(I))^(-1) bold(Y)_(:,j)
  - 1/2 log det (bold(K) + sigma^2 bold(I)) - N/2 log 2 pi.
$

The quadratic term measures agreement with $bold(Y)_(:,j)$ under the covariance model.
The log-determinant term accounts for the volume of output vectors to which that covariance assigns substantial density.
Together they balance agreement with $bold(Y)$ against covariance flexibility.
A prior on the hyperparameters changes the criterion from maximum likelihood to a posterior mode; the GP-EDM formulation of Munch, Poynor and Arriaza uses an ARD prior that favours negligible effects unless $bold(X)$ and $bold(Y)$ support them.
With $E' > 1$ this is one Gaussian process per column of $bold(Y)$.
Sharing the hyperparameters across the columns keeps a single smoother matrix, and with it the freedom to widen $bold(Y)$ at no cost; fitting them per column gives one smoother matrix per column instead, and that freedom is lost.

Because the fitted $phi_j$ determine input-coordinate relevance, kernel fitting also provides a way to assess candidate coordinates.
Specifically, one may supply a broad set of candidate variables and lags in $cal(J)$ and use the fitted $phi_j$ to identify coordinates that affect prediction of the chosen output.
This procedure selects predictors for that regression function.
As with ARD above, this selection does not establish injectivity of the delay-coordinate map.

Fitting these parameters also changes the linear-smoother property.
Once $eta$, $phi$ and $sigma^2$ are fitted from $bold(Y)$, the smoother matrix depends on $bold(Y)$, and the complete fitting procedure is not linear in $bold(Y)$.
The same dependence arises when $k$ or $theta$ is selected by predictive performance.
A higher-dimensional hyperparameter search increases the need for regularization and evaluation on pairs not used for selection.

= The three methods compared

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

    [Penalty], [none], [Tikhonov on $bold(C)_1$, strength $alpha$], [$sigma^2 norm(f)^2_(cal(H)_kappa)$],
    [Row of $bold(B)$],
    [$k$ nonzero, nonnegative, sums to $1$],
    [generally dense, signed, sums to $1$],
    [generally dense, signed, sum unconstrained],

    [Prediction], [in the convex hull of the targets], [unrestricted], [shrunk toward the prior mean],
    [Uncertainty from the fitted model], [not supplied], [not supplied], [predictive variance],
    [Solved per query],
    [$k$ neighbours and $k$ weights],
    [an $(E+1) times (E+1)$ system],
    [nothing; one $N times N$ solve serves all],
  )
]

= Designing input and output coordinates

An analysis specifies $cal(J)$ for the input and $cal(K)$ for the output.
$bold(X)$ and $bold(Q)$ use the same $cal(J)$ because their rows are compared in one input-coordinate space.
$bold(Y)$ uses $cal(K)$, whose variables, lags and dimension may differ from those of $cal(J)$.
Because these coordinate choices define the estimation problem, the same delay coordinates $cal(J)$ and $cal(K)$ can be used with any of the three estimators.

To illustrate these choices, the examples take two variables ($d = 2$).
An output horizon of $h$ is represented by an output lag of $-h$.

#block(breakable: false)[
  #table(
    columns: 4,
    stroke: 0.5pt + luma(180),
    inset: 6pt,
    table.header(
      [*Goal*],
      [*Input $cal(J)$ ($bold(X)$, $bold(Q)$)*],
      [*Output $cal(K)$ ($bold(Y)$)*],
      [*Sizes ($bold(X)$, $bold(Y)$)*],
    ),
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
This output is degenerate because $bold(Y)$ is the first column of $bold(X)$ and is already contained in the input vector.

*One variable to another.*
It asks whether the delay vectors built from variable 1 recover variable 2 at $h$ steps ahead, so a forecast in time and a map between variables are carried out in a single operation.
With $h = 0$ the row is *cross mapping*: the delay vectors built from one variable are asked for the contemporaneous value of another.
*Convergent cross mapping* (CCM) repeats this estimate over increasing training-set sizes and evaluates how cross-map skill changes with $N$.
Under the deterministic coupled-system assumptions of CCM, an effect variable contains information about the state of its causes.
Consequently, using variable 1 delay coordinates to recover variable 2 tests the direction variable 2 $arrow.r$ variable 1, not the reverse.
If variable 1 reconstructs the shared state, increasing the training set supplies closer analogues and the cross-map estimate of variable 2 improves before approaching its finite-noise limit.
Convergence is therefore evidence consistent with that causal direction only when common forcing, synchrony, temporal trends, sampling effects and alternative directions have been addressed#footnote[Sugihara, May, Ye, Hsieh, Deyle, Fogarty & Munch (2012), “Detecting causality in complex ecosystems,” _Science_ 338(6106):496--500.].

*Adding input coordinates.*
The number of training pairs is unchanged while the input-coordinate dimension grows from $2$ to $4$.
The new coordinates may separate states that previously looked identical, but they also make close neighbours harder to find in a finite training set unless the added coordinates are redundant on the sampled state set.

*Adding output coordinates.*
The input distances and neighbours are unchanged, and only the number of columns of $bold(Y)$ grows.
Several variables can therefore reuse one fixed smoother matrix when estimator parameters are shared across output coordinates.

*Different lags per variable.*
Since each delay coordinate selects an observation function and a lag, neither the number of delay coordinates nor the spacing of the lags has to agree across observation functions.
For example, the displayed delay coordinates read variable 1 at lags $0$ and $tau$ and variable 2 at lag $2 tau$ only.
This specifies delay coordinates that mix variables and time scales.
Generalized embedding theorems cover such delay coordinates when their regularity, dimension and periodic-orbit conditions hold.

*Several output horizons.*
Each coordinate of $cal(K)$ carries its own lag, so the horizons $h_1$ and $h_2$ are estimated at once.
With fixed estimator parameters, neighbours and smoother coefficients are determined by the input alone, so another horizon adds another output column.
Direct estimation does not feed earlier predictions into later inputs, so it avoids recursive propagation of earlier prediction errors.

For iterated forecasting, another choice is $cal(K) = cal(J)$ with every lag shifted by $-h$.
The columns of $bold(Y)$ then carry the input coordinates advanced by $h$ steps, so the predicted row has the coordinates required for another application of the same fitted map.
This alignment makes iterated multi-step forecasting possible.

The time indices also specify how predictions are evaluated: taking $M = N$ and $s_m = t_m$ for every $m$, and excluding each training pair from its own neighbourhood gives leave-one-out prediction.
A *Theiler window* widens the exclusion by removing every training index within $r$ sampling steps of the query index#footnote[Theiler (1986), “Spurious dimension from correlation algorithms applied to limited time-series data,” _Physical Review A_ 34(3):2427--2432, introduced temporal exclusion to prevent serial dependence from creating spurious near neighbours in correlation-dimension estimation.].
In prediction, this window prevents overlapping or strongly dependent delay vectors from serving as nominally independent validation cases.
The required width depends on the lags, forecast horizon, serial dependence and intended deployment setting.

= Assumptions and limits

The regression function remains meaningful beyond deterministic dynamics, but each interpretation requires its own assumptions.

- *Stable relationship*: the conditional mean of $U_cal(K)(t)$ given $U_cal(J)(t)$ must be the same $F$ at the training and query indices. A regime change can alter $F$ even when inputs remain numerically close.
- *Support*: a local estimate is supported only where the training set contains inputs near the query. A query outside the sampled support requires extrapolation, regardless of the estimator's algebraic ability to return a number.
- *Dynamical interpretation*: interpreting $F$ as $Psi_cal(K) compose Psi_cal(J)^(-1)$ requires deterministic dynamics and an input delay-coordinate map that embeds $A$. Otherwise $F$ is a conditional mean, and causal or structural language needs additional assumptions.
- *Temporal dependence*: ergodicity justifies long-run averages but supplies neither an effective sample size nor a convergence rate. Rates for local regression under temporal dependence require stronger conditions such as mixing. Validation splits and uncertainty calculations must preserve the temporal information available at the intended prediction time.
- *Leakage*: output times used for evaluation must be absent from fitting and parameter selection. Overlapping delay windows and temporally adjacent states can also make nominal train and test cases nearly identical. A Theiler window enforces separation when that separation matches the intended prediction task.
- *Coordinate scale*: Euclidean distance changes when one input coordinate is rescaled. Standardization, a physically chosen metric, or fitted kernel scales are modelling choices. ARD estimates relative predictive scales conditional on the units, kernel and hyperprior; it does not remove the need to state those choices.
- *Parameter selection*: choosing $cal(J)$, $cal(K)$, $k$, $theta$, $alpha$ or kernel parameters from prediction performance uses output information. Performance of the selected configuration requires evaluation pairs that played no part in that selection.
- *Dimension*: finite training sets become sparse as the effective dimension of the input distribution grows. For independent pairs whose inputs lie on a smooth lower-dimensional manifold, local polynomial regression can attain rates governed by the manifold dimension rather than the ambient coordinate dimension#footnote[Bickel & Li (2007), “Local polynomial regression on unknown manifolds,” _IMS Lecture Notes--Monograph Series_ 54:177--186.]. This theorem does not by itself establish an EDM rate on a fractal attractor with dependent pairs. Measurement noise can also spread the input delay vectors away from a lower-dimensional set, making the ambient dimension relevant in finite samples.
- *Skill measure*: Pearson correlation $rho$ is unchanged by a positive affine transformation of the predictions. It therefore does not detect additive bias or multiplicative miscalibration. Error and calibration measures are needed when their magnitudes matter.

The framework fixes the regression function, the three matrices and the estimator families.
An analysis additionally fixes the input coordinates, output coordinates, training indices, query indices, distance scale, tuning procedure and evaluation measure.
