default:
	nvcc NMF_gd.cu -lcusparse -lcublas -o NMF_gd
clean:
	rm NMF_gd
