from tgan.model import TGANModel
import os
import argparse

example_path = 'examples/'
model_path = 'models/'
model1 = 'model1'
model2 = 'model2'
model3 = 'model3'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic samples using pre-trained generator model')
    parser.add_argument('--model', type=str, default='model1')
    parser.add_argument('--num_samples', type=int, default=1400)
    parser.add_argument('--output_path', type=str, default='examples/model1.csv')
    args = parser.parse_args()
    
    new_tgan = TGANModel.load(os.path.join(model_path,args.model))
    new_samples = new_tgan.sample(args.num_samples)
    new_samples = round(new_samples).astype(int)
    new_samples.to_csv(args.output_path, index=False)