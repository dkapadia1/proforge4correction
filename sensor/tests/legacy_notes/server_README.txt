python server to analyze local images

possible ways to improve detection: increase vertical hessian scaling in region where gaussian detection finds it.

analyze_five : gaussian analysis
analyze_six : main centerline extraction + visualization of all methods
Also has filament confidence skew, a function that uses the gradients of the skews, a confidence array to calculate a reasonable place for the filament, then analyzes that region for split
Using flawed analysis methods of assuming all x < y is not split and split if else, this returns a residual of .62
server : simple webserver that takes path of a image and returns the deviation from the laserline

