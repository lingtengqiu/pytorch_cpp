#include<iostream>
#include<memory>
#include<assert.h>
#include<torch/script.h>
#include"aux.h"
int main(int argc,const char * argv[])
{
	if(argc != 2)
	{
		std::cerr<<"usage: example-app <path-model>"<<std::endl;
		return -1;

	}
	std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);
	std::vector<torch::jit::IValue> inputs;
	inputs.push_back(torch::ones({1,3,224,224}));
	at::Tensor output = module->forward(inputs).toTensor();
	output = output.cuda();
	std::cout<<output.slice(1,0,5)<<std::endl;
	assert(module!=nullptr);
	std::cout<<"ok"<<std::endl;

}
