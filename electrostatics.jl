using PyPlot

function julia(z, c; maxiter=50)
  for n = 1:maxiter
    if abs2(z) > 4
      return -1
      # -1 because we are conceiving it as _charges_
      # return n-1 if you want something pretty
    end
    z = z * z + c
  end
  return 1
  # return maxiter if you want something pretty
end

function shuffle_mat(mat)
  #### do it another way, maybe
  mat_shape = size(mat)
  first_range = 1:shape_ranges[0]
  second_range = 1:shape_ranges[1]
  shuffle(first_range)
  shuffle(second_range)
  mat
  #some idx shenanigans ####
end

function electropotential_energy(mat)
  # construe the matrix as a set of points in a lattice
  # possible optimization:
  # energy minimization is comparable to graph cut
  # graph cut very related to max flow
  # go use ford-fulkerson, or edmonds-karp in some way
  mat_shape = size(mat)
  energy = 0
  for i = 1:mat_shape[1]
    for j = 1:mat_shape[2]
      first_point = mat[i,j]
      for k = 1:mat_shape[1]
        for l = 1:mat_shape[2]
          second_point = mat[k,l]
          distance = sqrt((i - k) ^ 2 + (j - l) ^ 2)
          energy += (first_point * second_point) / distance
        end
      end
    end
  end
  energy
end

@time m = [ uint8(julia(complex(r,i), complex(-.06,.67))) for i=1:-.01:-1, r=-1.5:.01:1.5 ];

@time u_e = electropotential_energy(m)

#imshow(m, extent=[-1.5, 1.5, -1, 1])
#savefig("pics/julialang_shuffled")
