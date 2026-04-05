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
	std::vector<Tensor> qs;
	std::vector<Tensor> ks;
	std::vector<Tensor> vs;
	int d_k;
	Tensor gamma;
	Tensor beta;
	public:
	Transformer(int d_model, int max_len, int vocab_size, int seq_len, int num_heads,
	            std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs,
	            Tensor gamma, Tensor beta)
		: d_model(d_model),
		  max_len(max_len),
		  vocab_size(vocab_size),
		  seq_len(seq_len),
		  num_heads(num_heads),
		  qs(qs),
		  ks(ks),
		  vs(vs),
		  gamma(gamma),
		  beta(beta)
	{
		// d_k is the dimension per head
		this->d_k = d_model / num_heads;

		// Validate that we have the correct number of heads
		if (qs.size() != num_heads || ks.size() != num_heads || vs.size() != num_heads) {
			throw std::invalid_argument("Number of Q, K, V tensors must match num_heads");
		}
	}


	    Tensor attention(const Tensor& input, const Tensor& k, const  Tensor& q, const Tensor& v,int d_k){
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

		Tensor multiheadattention(std::vector<Tensor> qs, std::vector<Tensor> ks, std::vector<Tensor> vs, Tensor input,int d_k){
		std::vector<Tensor> res;
		for(int i =1;i<qs.size();i++){
		    res.push_back(attention(input,ks[i],qs[i],vs[i],d_k));
		}
		return attention(input,ks[0],qs[0],vs[0],d_k).concat_horizontal(res);
		}

		Tensor forward( Tensor input){
		    input = input.LayerNorm(gamma,beta);
		    input = input + multiheadattention(qs,ks,vs,input,d_k);
			return input;
		}
};
