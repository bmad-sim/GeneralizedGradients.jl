using BeamTracking, Beamlines, AtomicAndPhysicalConstants
using GeneralizedGradients, OffsetArrays
import Beamlines as bl

# 

fit, meta = read_gg_fit("gg_fit_result.h5")

function four_potential(x, y, s, t, p)
  B, A, dA = field_and_potential_evaluate_at(p, x, y, s) 
  potential = (0.0, A[1], A[2], A[3])
  jac = (0.0, 0.0, 0.0, 0.0,
         dA[1,1], dA[1,2], dA[1,3], 0.0,
         dA[2,1], dA[2,2], dA[2,3], 0.0,
         dA[3,1], dA[3,2], dA[3,3], 0.0)

  return potential, jac
end

species = Species("proton")
E0 = 220.94e9
p_over_q = bl.E_to_R(species, E0)
P0 = bl.E_to_pc(species, E0) / C_LIGHT
L = 0.055

ele = LineElement(four_potential = four_potential, four_potential_params = fit, 
          four_potential_normalized = false, L = L, 
          tracking_method=Yoshida(order=8, n_steps=60, radiation_damping_on=false))

v = [0.0 0.0 0.0 0.0 0.0 0.0]
q = [1.0 0.0 0.0 0.0]

B, A, dA = field_and_potential_evaluate_at(fit, v[1], v[3], 0.0)
v[2] = v[2] + chargeof(species) * A[1] / P0
v[4] = v[4] + chargeof(species) * A[3] / P0

b0 = Bunch(v, q, p_over_q_ref=p_over_q, species=species)
bline = Beamline([ele], p_over_q_ref=p_over_q, species_ref=species)
track!(b0, bline)

v2 = b0.coords.v
q2 = b0.coords.q

B, A, dA = field_and_potential_evaluate_at(fit, v2[1], v2[3], L)

v2[2] = v2[2] - chargeof(species) * A[1] / P0
v2[4] = v2[4] - chargeof(species) * A[3] / P0

println("$v2")