# Native G60 Cross-Flux Self-Alignment Register Theorem 041

## Theorem

The recovered real projective self-alignment census for

    r(f) = C(f)f - S4(f)f

decomposes under the native group of order 480 as

    40 = 10 + 15 + 10 + 5.

The four response classes are:

| Class | Size | Stabilizer | Rank | Operator norm | Status |
|---|---:|---:|---:|---|---|
| A10 | 10 | 48 | 6 | 1/3 | analytic_strict_local_maximum |
| N15 | 15 | 32 | 4 | sqrt((3+sqrt(5))/48) | self_aligned_nonextremal |
| N10 | 10 | 48 | 4 | 1/4 | self_aligned_nonextremal |
| N5 | 5 | 96 | 6 | sqrt(6)/12 | self_aligned_nonextremal |

The native ten-axis class is the unique recovered class with operator norm

    1/3.

It has squared pencil spectrum

    1/144 with multiplicity 4
    1/9 with multiplicity 2

and every native axis is an analytic strict local maximum with weakest quadratic drop coefficient

    1/18.

The non-axis classes have exact squared spectral laws:

    N15
    B(B^2 - B/8 + I/576) = 0

    N10
    B(B - I/16) = 0

    N5
    B = I/24

Thus self-alignment has four native response levels, but only the original ten-axis register is extremal among the recovered classes.

## Register notation

The original Thalean incidence Gram remains

    Q = M M^T

with M of shape 15 by 30.

The ten-axis four-flux line register remains

    R10 = L10 L10^T

with

    L10^T L10 = (5/2) I4
    R10^2 = (5/2) R10.

These are distinct Gram registers at distinct layers.

## Boundary

This theorem consolidates the recovered forty-line census and its exact orbitwise response laws.

It does not prove that the forty recovered lines are the complete real projective zero locus.

It does not yet prove the global inequality

    ||A(f)||_op <= (1/3)||f||

for every four-flux direction.

No physical energy, force, material, transport, or instability claim is made.
