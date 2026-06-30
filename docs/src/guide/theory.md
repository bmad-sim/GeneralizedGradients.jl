# How the GG calculation works

This page summarizes the mathematics behind `gg_fit`. The field-expansion table
is linear in the GG functions and their `s`-derivatives, which makes the fit a
(weighted) linear least-squares problem.

## The field expansion

For each field component $c \in \{B_x, B_y, B_s\}$,

```{math}
B_c(x,y,z) = \sum_{(n,m)} CS_{c,b}(n,m; x,y)\, b_{(n,m)}(z)
           + \sum_{(n,m)} CS_{c,a}(n,m; x,y)\, a_{(n,m)}(z)
           + \sum_{m}     CS_{c,bs}(m; x,y)\, bs_{(m)}(z),
```

where each coefficient is a polynomial in the transverse coordinates,

```{math}
CS_{c,f}(n,m; x,y) = \sum (\text{coeff}\cdot g_{\text{ref}}^{\,k}\, x^p\, y^q),
```

and the derivative towers are
$b_{(n,m)} = \mathrm{d}^m b_n/\mathrm{d}z^m$,
$a_{(n,m)} = \mathrm{d}^m a_n/\mathrm{d}z^m$, and
$bs_{(m)} = \mathrm{d}^{m+1} a_0/\mathrm{d}z^{m+1}$.

## Plane-by-plane least squares

The unknowns at a base plane $z_0$ are the function values and their derivatives
$f_{(n,m)}(z_0)$ for $m = 0 \ldots m_{\max}$. The field on a neighbouring plane at
offset $\mathrm{d}z = z - z_0$ is obtained by Taylor-extrapolating each
derivative,

```{math}
f_{(n,m)}(z_0 + \mathrm{d}z) = \sum_{j \ge m}
   \frac{\mathrm{d}z^{\,j-m}}{(j-m)!}\, f_{(n,j)}(z_0),
```

which makes the model linear in the base-plane unknowns. Each base plane is then
solved by weighted linear least squares over all field points lying within
`n_planes_add` planes of it. Adding planes (`m_max = 2·n_planes_add`) lets the
fit resolve higher derivatives and smooths the result; past some point, using
more planes makes the polynomial approximation *less* accurate, so there is an
optimum.

## The merit function and weights

For a base plane, the merit function minimized is

```{math}
\text{Merit} = \sum \text{weight}\,(B_{\text{table}} - B_{\text{GG}})^2 ,
```

with a per-point weight that factors as
$\text{weight}(x,y,\mathrm{d}z) = w_{\text{core}}(x,y)\, w_{\text{plane}}(\mathrm{d}z)$:

```{math}
w_{\text{core}}(x,y) = \texttt{core\_weight}\,
   \frac{r_{\max}^2}{r_{\max}^2 + r^2(\texttt{core\_weight}-1)},
\qquad r^2 = x^2 + y^2,
```

```{math}
w_{\text{plane}}(\mathrm{d}z) = 1 +
   (\texttt{outer\_plane\_weight}-1)\,\frac{|\mathrm{d}z|}{\mathrm{d}z_{\max}} .
```

`core_weight = 1` (the default) weights all transverse points equally; a value
> 1 favours near-axis points, which is usually desirable since a beam spends most
of its time near the core. `outer_plane_weight = 1` weights all planes equally;
a value below 1 (but non-negative) down-weights the outer planes. When
`n_planes_add = 0` (so $\mathrm{d}z_{\max} = 0$), $w_{\text{plane}}$ is taken to
be 1.

## Conversion to Bmad's convention

Bmad's `gen_grad_map` uses **azimuthal-harmonic** gradients $C_{m,\alpha}(z)$,
$\alpha \in \{\sin, \cos\}$, rather than this project's midplane-derivative GGs.
Equating the two expansions on the midplane gives exact recursions (solved in
order of increasing $m$):

```{math}
C^{[j]}_{m,s} = \frac{1}{m!}\Big[ b^{[j]}_m
   - (m-1)! \sum_{n\ge 1,\, m-2n\ge 1} W_n(m,n)\, C^{[j+2n]}_{m-2n,s} \Big],
```

```{math}
C^{[j]}_{m,c} = \frac{1}{m!}\Big[ a^{[j]}_m
   - (m-1)! \sum_{n\ge 1,\, m-2n\ge 1} W_c(m,n)\, C^{[j+2n]}_{m-2n,c}
   - [m \text{ even}]\,(m-1)!\,U_s(m)\, b_s^{[m+j-1]} \Big],
```

```{math}
C^{[j]}_{0,c} = b_s^{[j-1]} \quad (j \ge 1),
```

where $x^{[j]} \equiv \mathrm{d}^j x/\mathrm{d}s^j$ is supplied directly by the
fit. The full derivation, including the mixing weights $W_n$, $W_c$, and $U_s$,
is in the `write_bmad_gg_fit` docstring (see the **API Reference**).

## References

- S. Van der Schueren *et al.*, *"Magnetic Field Modelling and Symplectic
  Integration of Magnetic Fields on Curved Reference Frames for Improved
  Synchrotron Design: First Steps"* (copy in the `papers/` directory).
- D. Sagan, IPAC'23 — the Venturini–Dragt azimuthal-harmonic field expansion
  used by Bmad.
