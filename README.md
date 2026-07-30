# 46 Native G60 Hodge Mechanics

## Purpose

Project 46 develops a discrete Hodge and field-mechanics layer on the
native genus-21 surface established by Project 45.

Project 45 remains the authority for:

- the 60 native vertices
- the 120 native edges
- the 20 oriented dodecagonal faces
- the signed boundary matrices B1 and B2
- the identity B1 B2 = 0
- the genus-21 surface structure
- the native automorphism action
- the integral homology
- bounded canonicity inside the native successor-orbit universe

Project 46 imports those objects as immutable inputs. It does not
reconstruct, alter, or reinterpret the surface.

## Core question

Can the native symmetry of G60 determine an admissible discrete Hodge
structure and a lawful finite field mechanics on the genus-21 surface?

## Baseline cochain complex

The imported chain complex is

    C2 --B2--> C1 --B1--> C0

with dimensions

    dim C0 = 60
    dim C1 = 120
    dim C2 = 20

The cochain operators are

    d0 = B1^T
    d1 = B2^T

and satisfy

    d1 d0 = 0.

## First model

The first baseline uses uniform positive inner products:

    star0 = I60
    star1 = I120
    star2 = I20

This gives

    Delta0 = B1 B1^T

    Delta1 = B1^T B1 + B2 B2^T

    Delta2 = B2^T B2

No metric distance, area, impedance, permittivity, permeability, or
physical constant is inferred from these identity weights.

## First theorem target

Prove the exact orthogonal decomposition

    C1 = im(d0) direct_sum H1 direct_sum im(delta2)

with dimensions

    dim im(d0) = 59
    dim H1 = 42
    dim im(delta2) = 19.

Equivalently,

    59 + 42 + 19 = 120.

## Research ladder

1. Import and validate the Project 45 chain complex.
2. Construct the uniform Hodge baseline.
3. Prove the discrete Hodge decomposition.
4. Export a harmonic 1-form basis.
5. Verify native symmetry commutation.
6. Decompose the harmonic space under symmetry.
7. Audit gauge equivalence and curvature invariance.
8. Study static field response.
9. Introduce a bounded wave-operator baseline.
10. Verify discrete energy conservation.
11. Search constrained weighted Hodge candidates.
12. Package the theorem and boundary.

## Claim boundary

This project may establish:

- exact discrete Hodge decomposition
- native gauge invariance
- harmonic circulation sectors
- symmetry-compatible Laplacians
- conservative finite field dynamics
- bounded constitutive-weight candidates

This project does not initially establish:

- physical electromagnetism
- photons
- spacetime
- quantum mechanics
- gravity
- physical constants
- physical energy
- force laws
- simulation of the physical universe
- unification

## Working principle

Do not write Maxwell-looking equations and mistake notation for physics.

The question is whether native construction and symmetry constrain the
operators strongly enough that the mechanics protects itself.
