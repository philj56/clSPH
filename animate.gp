file = "out.dat"
tStep = 1;
tWidth = 5;

set xrange [-5:5]
set yrange [-5:5]

set grid
set term pngcairo size 1080, 1080
unset key

stats file
N = 500 #floor(system("wc -l \`pwd\`/out.dat")/100.)
do for [t=tWidth+1:N:tStep] {
	set output sprintf("Frames/%04d.png", (t-tWidth-1)/tStep + 1)
	plot for [p=2:STATS_columns:3] file every ::t-tWidth::t u p:p+1 w l
	print t
}
unset output
