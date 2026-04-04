#include"parser.hpp"
#include<stdexcept>
#include<map>
#include<string>
#include<iostream>


class Runner{
	private:
	std::map<std::string, Tensor> weights;
	public:
	Runner() {
		weights  = parse_weights();
	}
	void list(){
	for(auto it : weights) 
		std::cout<<it.first<<"\n";
	}

	void run() {
		// work here
	}
};


