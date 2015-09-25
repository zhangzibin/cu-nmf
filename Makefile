default:
	nvcc NMF_gd.c -lcusparse -lcublas -o NMF_gd
clean:
	rm NMF_gd
