using PyPlot

function julia(z, c; maxiter=50)
  for n = 1:maxiter
    if abs2(z) > 4
      return 0
      #return n-1 if you want pretty
    end
    z = z * z + c
  end
  return 1
  # return maxiter if you want pretty
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
  mat_shape = size(mat)
  for i in 1:mat_shape[0]
    for j in 1:mat_shape[1]
      for k in 1:mat_shape[0]
        for l in 1:mat_shape[1]
          # walla boinka doinka
        end
      end
    end
  end
end

@time m = [ uint8(julia(complex(r,i), complex(-.06,.67))) for i=1:-.002:-1, r=-1.5:.002:1.5 ];

@time m = shuffle_mat(m)

imshow(m, extent=[-1.5, 1.5, -1, 1])
savefig("pics/julialang_shuffled")
