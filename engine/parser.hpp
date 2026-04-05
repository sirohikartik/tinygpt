
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
			std::string filename = file.path().filename().string();
			std::string name = filename.substr(0, filename.size() - 4);
			weights[name] = Tensor(v,rows,cols);
			std::cout<<name<<"\n";
			std::cout<<"Done weight "<<i<<" shape :"<<rows<<" "<<cols<<"\n";
			i++;
		}
		catch(std::exception& e) {
			std::cout<<"Found mask skipping \n";
		}
	}
	return weights;
}
