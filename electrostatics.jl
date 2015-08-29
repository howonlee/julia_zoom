
function electrostatics_energy(arr)
  arr_size = shape(arr)
  energy = 0
  for i in arr_size[0] #### range iterator or something
    for j in arr_size[1]
      for k in arr_size[0]
        for l in arr_size[1]
          energy += something #######3
        end
      end
    end
  end
  energy
end

function create_fractal()
  ####
end

function scramble_fractal(arr)
  scrambled
end

function unscramble(frac, num_iters)
  unscrambled_frac = copy(frac)
  for i in range(num_iters)
    # do the optimization
  end
  unscrambled_frac
end

frac = create_fractal()
scrambled_fract = scramble_fractal(frac)
unscrambled_frac = unscramble(scrambled_frac)
# save the unscrambled_frac to file
