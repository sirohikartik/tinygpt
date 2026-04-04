#include"parser.hpp"
#include<map>
#include<string>
#include<iostream>
#include<cmath>

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

class Transformer{
	int d_model;
	int max_len;
	int vocab_size;
	int seq_len;
	int num_heads;
	public:

	    Tensor attention(const Tensor& input, const Tensor& k, const  Tensor& q, const Tensor& v,float d_k){
					d_k = std::sqrt(d_k);
					Tensor Q = input * q;
					Tensor K = input * k;
					Tensor V = input * v;
					Q = Q * K.t();
					Q  = Q/d_k;
					Q = Q.mask();
					Q = Q.softmax();
					Q = Q * V;
					return Q;
		}

		Tensor multiheadattention(){}

		Tensor forward(Tensor& input){

		}
};
