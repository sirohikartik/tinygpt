#include <stdexcept>
#include <vector>
#include <algorithm>
#include"npy.hpp"
#include<string>
#include<filesystem>
#include"tensor.hpp"
#include<iostream>
#include<map>

namespace fs = std::filesystem;

std::string path_header = "weights";
std::map<std::string,Tensor> parse_weights(){
	std::map<std::string,Tensor> weights;

	int i = 0;
	for(const auto& file : fs::directory_iterator(path_header)){
		try{
		npy::npy_data d = npy::read_npy<float>(file.path().string());
			std::vector<unsigned long> shape = d.shape;
			std::vector<float> v = d.data;
			int rows = static_cast<int>(shape[0]);
int cols = shape.size() > 1 ? static_cast<int>(shape[1]) : 1;
			weights[file.path().string()] = Tensor(v,rows,cols);
			std::cout<<file.path().string()<<"\n";
			std::cout<<"Done weight "<<i<<" shape :"<<shape[0]<<" "<<shape[1]<<"\n";
			i++;
		}
		catch(std::exception& e) {
			std::cout<<"Found mask skipping \n";
			}
		}
	return weights;
}
