default:
	nvcc NMF_gd.cu -lcusparse -lcublas -o NMF_gd
	nvcc NMF_pgd.cu -lcusparse -lcublas -o NMF_pgd
clean:
	rm NMF_gd
	rm NMF_pgd
