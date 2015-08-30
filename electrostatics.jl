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

@time m = [ uint8(julia(complex(r,i), complex(-.06,.67))) for i=1:-.002:-1, r=-1.5:.002:1.5 ];

imshow(m, extent=[-1.5, 1.5, -1, 1])
savefig("pics/julialang_julia")
