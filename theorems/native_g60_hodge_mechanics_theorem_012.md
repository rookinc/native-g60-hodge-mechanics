# Native G60 Hodge Mechanics Theorem 012

## Status

- theorem_pass: `true`
- verdict: `native_g60_finite_hodge_mechanics_theorem_packaged`

## Theorem

On the imported native genus-21 surface of AT4val[60,6], the full native automorphism group preserves a positive uniform discrete Hodge complex whose edge-cochain space decomposes orthogonally into exact, harmonic, and coexact subspaces of dimensions 59, 42, and 19. The harmonic subspace carries a faithful 42-dimensional group representation with irreducible dimensions 2, 6, 6, 8, 10, and 10. Gauge equivalence removes exactly the 59-dimensional exact sector, leaving a 61-dimensional quotient formed by 42 harmonic and 19 coexact directions. The edge Laplacian has rank 78 and harmonic kernel dimension 42, admits the stated static solvability condition, and defines a finite homogeneous second-order dynamics with a conserved quadratic invariant. Full native symmetry forces every positive diagonal Hodge structure to have the form star0=aI60, star1=bI120, star2=cI20, while leaving the two effective positive ratios b/a and c/b undetermined.

## Exact finite structure

\[
C^0 \\xrightarrow{d_0} C^1 \\xrightarrow{d_1} C^2,
\\qquad
d_1d_0=0,
\]

with

\[
\\dim C^0=60,
\\qquad
\\dim C^1=120,
\\qquad
\\dim C^2=20.
\]

For the uniform positive Hodge baseline,

\[
C^1
=
\\operatorname{im}d_0
\\oplus
\\ker\\Delta_1
\\oplus
\\operatorname{im}d_1^{\\mathsf T},
\]

and

\[
120=59+42+19.
\]

## Harmonic symmetry

The native group acts faithfully on the harmonic sector, with concrete
irreducible dimensions

\[
42=2+6+6+8+10+10.
\]

The central half-turn splits the harmonic space as

\[
42=12_+ + 30_-.
\]

## Gauge quotient

\[
C^1/\\operatorname{im}d_0
\\cong
\\mathcal H^1
\\oplus
\\operatorname{im}d_1^{\\mathsf T},
\]

with

\[
61=42+19.
\]

Face curvature is gauge invariant but does not classify the harmonic
part of the gauge quotient.

## Static and finite dynamics

The static equation

\[
\\Delta_1A=J
\]

is solvable precisely when

\[
P_{\\mathrm{harmonic}}J=0.
\]

The bounded second-order system is

\[
\\ddot A+\\Delta_1A=J(t).
\]

For the homogeneous system,

\[
\\frac{dE}{dt}=0.
\]

For the forced system,

\[
\\frac{dE}{dt}
=
\\langle\\dot A,J\\rangle.
\]

## Symmetry-compatible diagonal family

Full native symmetry forces

\[
\\star_0=aI_{60},
\\qquad
\\star_1=bI_{120},
\\qquad
\\star_2=cI_{20},
\]

with

\[
a>0,
\\qquad
b>0,
\\qquad
c>0.
\]

The operators depend only on

\[
x=\\frac ba,
\\qquad
y=\\frac cb.
\]

Native symmetry determines the form of the diagonal mechanics, but not
the numerical point inside this positive two-ratio family.

## Boundary

The theorem concerns a finite combinatorial Hodge mechanics. It does not derive a metric geometry, physical units, constitutive ratios, electromagnetism, photons, spacetime, quantum mechanics, gravity, physical forces, physical energy, a simulation of the physical universe, or a unification.

## Next frontier

Classify the full positive self-adjoint commutant beyond diagonal
weights, then test whether native construction data selects the free
ratios \(x\) and \(y\).
