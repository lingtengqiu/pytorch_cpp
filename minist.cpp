#include<torch/script.h>
#include<cstddef>
#include<iostream>
#include<string>
#include<vector>
#include<torch/torch.h>
#include<assert.h>
#include<typeinfo>
struct Net: torch::nn::Module
{
	Net():conv1(torch::nn::Conv2dOptions(1,10,5)),conv2(torch::nn::Conv2dOptions(10,20,5)),fc1(320,50),fc2(50,10)
	{
		register_module("conv1",conv1);
		register_module("conv2",conv2);
		register_module("fc1",fc1);
		register_module("fc2",fc2);	
	}
	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(torch::max_pool2d(conv1->forward(x),2));
		x = torch::relu(torch::max_pool2d(conv2->forward(x),2));

		x = x.view({-1,320});
		x = torch::relu(fc1->forward(x));
		//x = torch::dropout(x,0.5,is_training());
		x = fc2->forward(x);
		return torch::log_softmax(x,1);
	}
	torch::nn::Conv2d conv1;
	torch::nn::Conv2d conv2;
	torch::nn::Linear fc1;
	torch::nn::Linear fc2;
};

//this part is using the options about this system


struct Options{
	std::string data_root{"data"};
	int32_t batch_size{64};
	int32_t epochs{10};
	double lr{0.045};
	double momentum{0.9};
	bool no_cuda{false};
	int32_t seed{1};
	int32_t test_batch_size{1000};
	int32_t log_interval{10};
};
/*
template<typename DataLoader>
void train(int32_t epoch,const Options& options,Net & model, torch::Device device,DataLoader & data_loader,
			torch::optim::SGD & optimizer,size_t dataset_size){
			
			//step->1 begin train
			model.train();

			size_t batch_idx = 0;

			//step->2 iter
			
			for(auto &batch: data_loader)
			{
				auto data =batch.data.to(device);
				auto target = batch.target.to(device);

				optimizer.zero_grad();
				auto output = model.forward(data);
				auto loss = torch::nll_loss(output,target);
				loss.backward();
				optimizer.step();
			//step->3 print the output
				if(batch_idx ++ % options.log_interval == 0)
				{
					std::cout<<"Train Epoch: "<<epoch<<"["<<batch_idx*batch.data.size(0)<<"/"
						<<dataset_size<<"]\tloss: "<<loss.template item<float>()<<std::endl;
				}

			}
}
*/
template <typename DataLoader>
void train(
    int32_t epoch,
    const Options& options,
    Net& model,
    torch::Device device,
    DataLoader& data_loader,
    torch::optim::SGD& optimizer,
    size_t dataset_size) {
  model.train();
	model.to(device);
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
    optimizer.zero_grad();
    auto output = model.forward(data);
		std::cout<<output[0]<<std::endl;
    auto loss = torch::nll_loss(output, targets);
    loss.backward();
    optimizer.step();

    if (batch_idx++ % options.log_interval == 0) {
      std::cout << "Train Epoch: " << epoch << " ["
                << batch_idx * batch.data.size(0) << "/" << dataset_size
                << "]\tLoss: " << loss.template item<float>() << std::endl;
    }
  }
}
template <typename DataLoader>
void test(
    std::shared_ptr<torch::jit::script::Module> &model,
    torch::Device device,
    DataLoader &data_loader,
    size_t dataset_size) {
  torch::NoGradGuard();
  double test_loss = 0;
  int32_t correct = 0;
  for (const auto& batch : data_loader) {
    auto data = batch.data.to(device), targets = batch.target.to(device);
		std::vector<torch::jit::IValue> inputs;
		inputs.push_back(data);
		at::Tensor output = model->forward(inputs).toTensor();
		output = torch::log_softmax(output,1);
    test_loss += torch::nll_loss(
                     output,
                     targets,
                     {},
                     Reduction::Mean)
                     .template item<float>();
    auto pred = output.argmax(1);
    correct += pred.eq(targets).sum().template item<int64_t>();
  }

  test_loss /= dataset_size;
  std::cout << "Test set: Average loss: " << test_loss
            << ", Accuracy: " << correct << "/" << dataset_size << std::endl;
}

struct Normalize : public torch::data::transforms::TensorTransform<> {
  Normalize(float mean, float stddev): mean_(torch::tensor(mean)), stddev_(torch::tensor(stddev)) {}
	//here has a trap must be carefully deal with ---> so in here  the fuction sub_ will change the original so here must
	//using the sub method
  torch::Tensor operator()(torch::Tensor input) 
	{
	    return input.sub(mean_).div(stddev_);
	}
	torch::Tensor mean_, stddev_;
};
auto main(int argc, const char * argv[])->int{
	torch::manual_seed(0);
	Options options;
	options.data_root = argv[1];
	std::cout<<"data_root using training locates in "<<argv[1]<<std::endl;

	torch::DeviceType device_type;
	  if (torch::cuda::is_available() && !options.no_cuda) {
	    std::cout << "CUDA available! Training on GPU" << std::endl;
	    device_type = torch::kCUDA;
	  } else {
	    std::cout << "Training on CPU" << std::endl;
	    device_type = torch::kCPU;
	  }
	  torch::Device device(device_type);
	  std::shared_ptr<torch::jit::script::Module> le_net = torch::jit::load(argv[2]);
		le_net->to(device);
		
		assert(le_net !=nullptr);
		std::cout<<"OK!"<<std::endl;
	  auto train_dataset =
	      torch::data::datasets::MNIST(
		  options.data_root, torch::data::datasets::MNIST::Mode::kTrain)
		  .map(Normalize(0.1307, 0.3081))
		  .map(torch::data::transforms::Stack<>());
	  const size_t dataset_size = train_dataset.size().value();
		std::cout<<dataset_size<<std::endl;

	  auto train_loader = torch::data::make_data_loader(
	      std::move(train_dataset), options.batch_size);

	  auto test_dataset = torch::data::datasets::MNIST(
		  options.data_root, torch::data::datasets::MNIST::Mode::kTest)
		  .map(Normalize(0.1307, 0.3081))
		  .map(torch::data::transforms::Stack<>());
		const size_t test_size = test_dataset.size().value();
	  auto test_loader = torch::data::make_data_loader(
				test_dataset,options.batch_size);
	  auto test_loader1 = torch::data::make_data_loader(
				test_dataset,options.batch_size);
	  //torch::optim::SGD optimizer(
	  //    net.parameters(),
	  //    torch::optim::SGDOptions(options.lr).momentum(options.momentum));
	    test(le_net, device, *test_loader, test_size);
	    test(le_net, device, *test_loader, test_size);
		
	  //for (int32_t epoch = 1; epoch <= 10; ++epoch) {
		//	train(epoch,options,net, device, *train_loader,optimizer, dataset_size);
	  // test(le_net, device, *test_loader1, test_size);
	  // }
}
