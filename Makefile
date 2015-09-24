default:
	nvcc NMF_sgd.c -lcusparse -lcublas -o NMF_sgd
clean:
	rm NMF_sgd
