file = "out.dat"

set xrange [-1:1]
set yrange [-1:1]
set zrange [-1:1]

set grid
unset key

set xlabel "x"
set ylabel "y"
set zlabel "z"

stats file
set ticslevel 0
splot for [p=2:STATS_columns:3] file every ::490::490 u p:p+1:p+2 w p pt 7

