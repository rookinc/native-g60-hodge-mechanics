# TOM-QM-027 v0.2.0 · Writing a reversible return record

November 20, 2026 · Project 46

A second G60 register can retain the distinction between the two four-contact return histories after the original register is restored. In the explicit two-register model, the reference history leaves the pointer at its ready vertex. The other history moves it by the native central involution $a$. The movement follows from the two route lifts and their reference correction. The bare G60 graph still needs a source-defined mechanism to couple the same route to two independently addressable carriers.

## The write operation

Use the four-contact witness on G15 sites $(0,1,3,10)$. Let $w_0$ be its seven-edge reference history with return $ab$, and let $w_1$ be its eight-edge history with return $b$. Prepend the flat triangle $(0,1,11,0)$ to the shorter route and the flat retrace $(0,1,0)$ to the longer one. The resulting histories have ten edge traversals each and their holonomies are unchanged. Ten is minimal for this padding rule because the graph has zero self-loops. Equal edge counts control one discrete timing variable. The padding and the interior paths are distinct parts of the recorded itineraries.

Start with system lift $x$ and pointer lift $m$ in the fiber at the first contact. The system and pointer use the same native edge map at each step of the selected route. Its holonomy $H_j$ acts on both. Then reverse the fixed reference route on the pointer and reverse the selected route on the system. The full map is

$$
(j,x,m)\longmapsto (j,x,H_0^{-1}H_jm).
$$

For $j=0$ the pointer is unchanged. For $j=1$, $H_0^{-1}H_j=(ab)^{-1}b=a$. The code composes native edge maps to compute the output. The deck label $a$ verifies the result. The inverse runs the three transports in reverse order, with the route choice $j$ retained.

| Selected history | System after return | System after reverse replay | Pointer after correction | Pointer at the later contact |
| --- | ---: | ---: | ---: | ---: |
| Reference, $ab$ | 8 | 0 | 0 | 25 |
| Alternative, $b$ | 15 | 0 | 40 | 9 |

The example starts both registers at G60 vertex 0. The audit checks both alternatives, all four system and pointer vertices over the first contact, and all 480 automorphisms. All 15,360 cases reproduce the write, its inverse and later discrimination. Conjugating both routes can exchange $b$ and $ab$, while their relative element $a$ is unchanged. Every subsequent common lift is a bijection and preserves the difference. A pointer following a fixed branch-independent itinerary cannot record the system's choice.

## Which carriers preserve the record

The calibrated two-state pointer is the pair $(m,am)$ inside a four-state fiber. A two-state quotient of the whole fiber partitions it into two pairs, one partition for each nonidentity deck shift. At every site, the stabilizer fixes only the partition defined by $a$. This quotient identifies the two pointer states. The other two quotients retain the flip but are exchanged by the full group.

The pinned Project 42 G30 certificate identifies $m$ and $am$ for every vertex. The pinned Project 41 history groupoid has $a$ in all four isotropy kernels and acting trivially on all four phase cosets. The four-flux representation also sends $a$ to the identity, as proved in v0.1.0. These are exact erasure controls for the new pointer. Recording the bit requires an interface that distinguishes the two calibrated vertices.

A shared closed route acts diagonally on the pair of lifts. It becomes identity in the Cartesian pair quotient $(G60\square G60)/\operatorname{diag}(V_4)$. An independently routed comparison preserves a relative displacement in that quotient. The quotient has 900 vertices and 3,600 edges, with four triangles at every vertex. The pinned G900 static body has the same size and degree, with two triangles at 180 vertices, three at 540 and four at 180. This refutes an isomorphism to the tested pair-quotient host.

## The conditional quantum instrument

After admitting the two-register model, take the two route choices as an orthonormal logical basis, and use the complex span of the four native pointer vertices as the pointer space. The audited permutation extends linearly to the unitary

$$
U=|0\rangle\langle0|\otimes I + |1\rangle\langle1|\otimes U_a.
$$

With the pointer prepared at vertex $m$, its two possible records are $|m\rangle$ and $|am\rangle$. The two nonzero Kraus operators on the route register are $K_j=|j\rangle\langle j|$. Their effects sum to identity. The resulting maps $\mathcal I_j(\rho)=K_j\rho K_j^\dagger$ are completely positive and trace nonincreasing. With route amplitudes $3/5$ and $e^{i\phi}4/5$, the instrument gives probabilities $9/25$ and $16/25$ via the usual Born rule. The measured probabilities are conditional on this preparation and readout model. [Watrous, *The Theory of Quantum Information*, Chapter 2](https://johnwatrous.com/wp-content/uploads/TQI.pdf)

For $\phi=0,\pi$, the route's $X=+1$ probability is $49/50$ or $1/50$ before registration. It is $1/2$ for either preparation when the pointer is unread. A joint observable $X\otimes U_a$ still distinguishes the two preparations with expectations $+24/25$ and $-24/25$. The corresponding classical mixture has zero expectation. The $Y\otimes U_a$ test detects phases $\pm\pi/2$ in the same way. Reversing the gate recovers the original coherent input. When the pointer's result is copied onto a further register, reversing only the gate leaves that further record and the route stays locally dephased. Preparing the pointer in an eigenstate of $U_a$ gives no distinguishable record. Its eigenvalue can instead produce a phase change on the route.

The exact instrument concerns the explicit route-controlled model. A native interaction that distributes the route to two physical carriers, permits independent reversal, and determines the available observables remains open. Objective collapse remains open.

## Replay

Use the Project 46 checkout with its existing NumPy and SymPy dependencies. The optional inputs reproduce the independent G900, G30 and Project 41 controls. Their canonical paths, revisions and hashes are in the [input lock](inputs.lock.json). The prior contact and flux source inputs retain the v0.1.0 hashes.

```sh
audit=audits/TOM-QM-027/v0.2.0
out=/tmp/tom-qm-027-readout
python "$audit/audit_readout.py" --g900-static-body /path/to/g900_static_body.v1.json \
  --g30-certificate /path/to/project42_g60_to_g30_a_quotient_certificate_035.json \
  --history-groupoid /path/to/project41_registered_history_transformation_groupoid_201h874a.v1.json \
  --output "$out"
python "$audit/audit_instrument.py" --output "$out"
```

See the [readout receipt](receipts/readout.json) and [conditional instrument receipt](receipts/conditional-instrument.json).
