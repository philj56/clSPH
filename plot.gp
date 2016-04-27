file = "out.dat"

set xrange [-2:2]
set yrange [-2:2]
set zrange [-2:2]

set grid
unset key

set xlabel "x"
set ylabel "y"
set zlabel "z"

stats file
set ticslevel 0
splot for [p=2:STATS_columns:3] file u p:p+1:p+2 w l
