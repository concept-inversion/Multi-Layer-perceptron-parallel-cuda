test: gpu_perceptron.cu
	nvcc gpu_perceptron.cu -o perceptron
	./perceptron
