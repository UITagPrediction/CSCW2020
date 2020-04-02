import torch
import argparse
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, Resize, Compose
import model
from visualization.core import Weights
from visualization.core.utils import image_net_preprocessing
from utils import module2traced,run_vis_plot

parser = argparse.ArgumentParser(description='All layers Visualise')
parser.add_argument('--tag', type=str, default='food',
                    help='model to visualise (default: food)')
parser.add_argument('--i', type=str, default='nan',
                    help='image to visualise (default: nan)')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image loader
image=Image.open(args.i)
image = image.convert('RGB')
input  = Compose([Resize((224,224)), ToTensor(), image_net_preprocessing ])(image).unsqueeze(0)
input = input.to(device)

# model
model_name = "resnet"
feature_extract = True
num_classes = 2
model, _ = model.initialize_model(model_name, feature_extract, num_classes, use_pretrained = True)
model = model.to(device)

def test():
	MODEL_PATH = 'backup/'+args.tag+'/checkpoint.pt'
	if device==torch.device('cpu'):
		model.load_state_dict(torch.load(MODEL_PATH,map_location='cpu'))
	else:
		model.load_state_dict(torch.load(MODEL_PATH))
	model.eval()

	model_traced = module2traced(model, input)
	first_layer = model_traced[0]
	vis = Weights(model, device)
	run_vis_plot(vis,input,first_layer, ncols=8)
	plt.show()

if __name__ == "__main__":
    test()