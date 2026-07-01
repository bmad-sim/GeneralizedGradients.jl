using BeamTracking, Beamlines

# 

function four_potential(x, y, s, t, p)
  potential = (0.0, 0.0, 0.0, 0.0)
  vec_potential = 
  return potential, jac
end

ele = LineElement(four_potential = four_potential, four_potential_params = , 
          four_potential_normalized = false, L = , 
          tracking_method=Yoshida(order=8, n_steps=60, radiation_damping_on=false))

species = Species("electron")
p_over_q = E_to_R(Species
v = [0.0 0.0 0.0 0.0 0.0 0.0]
q = [1.0 0.0 0.0 0.0]

b0 = Bunch(v, q, p_over_q_ref=p_over_q, species=species)
bl = Beamline([ele], p_over_q_ref=p_over_q, species_ref=species)
track!(b0, bl)
