
function electrostatics_energy(arr)
  arr_size = shape(arr)
  energy = 0
  for i in arr_size[0]
    for j in arr_size[1]
      for k in arr_size[0]
        for l in arr_size[1]
        end
      end
    end
  end
  energy
end

function create_fractal()
  ####
end

function unscramble()
  ####
end

frac = create_fractal()
unscrambled_frac = unscramble(frac)
# save the unscrambled_frac to file
