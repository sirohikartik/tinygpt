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

		Tensor multiheadattention(std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs, std::vector<Tensor> input,float d_k){
		std::vector<Tensor> res;
		for(int i =1;i<qs.size();i++){
		    res.push_back(attention(input[i],ks[i],qs[i],vs[i],d_k));
		}
		return attention(input[0],ks[0],qs[0],vs[0],d_k).concat_horizontal(res);
		}
};
