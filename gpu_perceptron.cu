#include <stdio.h>
#include <stdlib.h>
//#include "wtime.h"
#include "reader.hpp"

__global__ void
good_multiplication(int*a,int*b,int *c,int n_row,int n_comm)
{
    
    
__shared__ int intermediate[512];
int tid= threadIdx.x + blockIdx.x * blockDim.x;
int total_blocks= gridDim.x;
int block_step= n_row/total_blocks;
if(block_step<0){block_step=1;}
int block_start= blockIdx.x *block_step;
int block_stop= (blockIdx.x+1)*block_step;
if(blockIdx.x==total_blocks-1){block_stop=n_row;}
//printf("blockstep:%d,blockstart%d",block_step,block_start);
while(block_start<block_stop)
{
   __syncthreads();
    int length = n_comm; 
    //printf("\n For each block:blockId: %d, start:%d ,stop:%d\n ",blockIdx.x,block_start,block_stop);
    int block_offset = block_start*n_comm;
    int step = length/blockDim.x; //asumes threads are always less than length
    int start=threadIdx.x*step+block_offset;
    int end=(threadIdx.x+1)*step+block_offset;
    if(threadIdx.x==(blockDim.x-1)){end=(block_start+1)* length;}
    //printf(" start:%d,  step: %d, end: %d\n",start,step,end);

    while(start<end)
    {
        
       intermediate[start-block_offset]= a[start]*b[start-block_offset];
        //printf(" BlockID:%d,  row: %d, index: %d, value:%d,a:%d, b:%d \n ",blockIdx.x,block_start,start,intermediate[start-block_offset],a[start],b[start-block_offset]);
        start+=1;
    }
    __syncthreads();

    // Now add all the item in intermidate value with parallel reduce. Determine the number of steps required for reduction for each block.
    // start a loop 
    int depth=0;
    length=n_comm;
    while((threadIdx.x<(length/2)))
    {  
        depth+=1;
        int required= length/2+ length%2;
        int step= required/blockDim.x;
        if(step==0){step=1;}
        //printf("\n Step: %d, Required:%d , blockDim:%d\n  ",step,required,blockDim.x);
        int start= threadIdx.x *step; 
        int stop= (threadIdx.x+1)*step;
        int final_thread=0;
        if(required > blockDim.x)
        {final_thread=blockDim.x;}
        else
        {final_thread=required;}
        if(threadIdx.x==(final_thread-1))
            {
                stop =( length/2) +length%2;
            }
       
        //printf("ThreadID: %d, start: %d, stop: %d \n",threadIdx.x,start,stop);
        while(start<stop)
        {
           
            //printf(" Thread: %d   Start: %d\n",threadIdx.x,start);
            if(start==length/2)
            {
               
                if(length%2==1){intermediate[start]=intermediate[start+length/2];}
            
            }
            else {intermediate[start]+= intermediate[start+length/2];}
           
            if(depth==8)
            {
            //printf("Depth:%d,row:%d,Thread:%d,step:%d,length:%d, start:%d,access:%d. End:%d Intermediate value is %d\n",depth,block_start,threadIdx.x,step,length,start,length/2+start,stop, intermediate[start]);
            }
            start+=1;
            
        }
        
        length=required;
        //printf("\n ThreadId:%d  step:%d  start:%d   stop:%d\n",threadIdx.x,step,start,stop);
         
        if(length==1 && threadIdx.x==0 )
        {
        // write to global memory
        
        c[block_start]=intermediate[threadIdx.x];
        //printf("\n Thread 0, Block %d, wrote result %d, In:%d \n",blockIdx.x,c[block_start],block_start);
        break; 
        }
    
    }

block_start+=1;
__syncthreads();
}
}

__global__ 
void perceptron(float *train, float *test,float *d_w, float b)
{
	int dimension=61;
	printf("start perceptrion\n");
	// initiate weight and bias
	int epoch=1;float predict=0;
	b=0;
	for(int i=threadIdx.x;i<61;i+=blockDim.x)
	{
	d_w[i]=0;
	}
	int row=blockDim.x;
	int value=threadIdx.x;
	int expected=0;
	// for each epoch
	while (row<173)
	{
	
		//predict the label of the output
		for(i=threadIdx.x;i<61;i+=1)
		{
		predict = train[row*dimension+value]*d_w[value]+b;	
		}
	
	if(predict<0)
	{predict=-1;}
	else predict=0;
printf("Predict:\n");	
	//	
}
   
void display(int *a,int len)
{
    printf("\n");
    for (int i=0;i<len;i++)
    {
        printf("%d \n",a[i]);
    }
}

int main(void)
{
	float *train_data;
	int train_count;

	float *test_data;
	int test_count;
	float *W,b;
	int w_size= 61;
	reader("train_data.bin",train_data,train_count);
	reader("test_data.bin",test_data,test_count);
	
    float *d_w;
    float *d_train;
    float *d_test;
    cudaMalloc ((void **)&d_train,sizeof(float)*train_count);
    cudaMalloc ((void **)&d_test,sizeof(float)*test_count);
    cudaMalloc ((void **)&d_w,sizeof(int)*w_size);
    //printf("\n malloc worked");
   // double time_start=wtime(); 
    cudaMemcpy(d_train,train_data,sizeof(float)*train_count,cudaMemcpyHostToDevice);
    cudaMemcpy(d_test,test_data,sizeof(float)*test_count,cudaMemcpyHostToDevice);
     //cudaMemcpy(d_w,W,sizeof(int)*w_size,cudaMemcpyHostToDevice);
    //start_time=0;
    perceptron<<<1,1>>>(d_train,d_test,d_w,b);
    //End_time=0;
    cudaDeviceSynchronize();
    //cudaMemcpy(W,d_w,sizeof(int)*len, cudaMemcpyDeviceToHost);
    //double total_time= wtime()-time_start;

    //printf("After:");
    //display(c,512);
    //printf("\n Total_time= %f\n",total_time);
    return 0;
}
