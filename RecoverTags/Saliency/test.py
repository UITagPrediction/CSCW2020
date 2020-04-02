import os
import argparse
import torch
import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import model
from utils import *
from visualization.core import *
from visualization.core.utils import image_net_postprocessing
from visualization.core.utils import image_net_preprocessing
from torchvision.transforms import ToTensor, Resize, Compose, ToPILImage

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description='Data Visualise')
parser.add_argument('--tag', type=str, default='food',
                    help='model to visualise (default: food)')
args = parser.parse_args()

model, _ = model.initialize_model("resnet", True, 2, use_pretrained = True)
model = model.to(device)

def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = ToPILImage()(image)
    return image

def test_model(SAVE_MODEL_PATH,TEST_PATH='./Data/',SAVE_PATH='./OUT/'):
	# init
	if device==torch.device('cpu'):
		model.load_state_dict(torch.load(SAVE_MODEL_PATH,map_location='cpu'))
	else:
		model.load_state_dict(torch.load(SAVE_MODEL_PATH))
	print('#'*30)
	print('Backup loaded')
	print('#'*30)
	model.eval()
	vis = SaliencyMap(model, device)
	if os.path.isdir(SAVE_PATH):
		from shutil import rmtree
		rmtree(SAVE_PATH)
	os.mkdir(SAVE_PATH)

	# load images
	images_path = os.listdir(TEST_PATH)
	if os.path.exists(TEST_PATH+'.DS_Store'):images_path.remove('.DS_Store')
	images = list(map(lambda x: Image.open(TEST_PATH+x), images_path))
	inputs  = [Compose([Resize((224,224)), ToTensor(), image_net_preprocessing])(x).unsqueeze(0) for x in images]  # add 1 dim for batch
	inputs = [i.to(device) for i in inputs]

	# visualise images
	for i,input in enumerate(inputs):
		print(i,'processing .....')
		print('#'*30)
		model_traced = module2traced(model, input)
		first_layer = model_traced[0]
		out, info = vis(input, first_layer,guide=True)
		
		origin_img = images[i]
		feature_img = tensor_to_PIL(out)
		fig = plt.figure()
		ax1 = fig.add_subplot(1, 2, 1)
		ax1.set_title("origin")
		ax1.imshow(np.asarray(origin_img))
		plt.axis('off')
		ax2 = fig.add_subplot(1, 2, 2)
		ax2.set_title("feature")
		ax2 = plt.imshow(np.asarray(feature_img))
		plt.axis('off')
		plt.savefig(SAVE_PATH+images_path[i])

if __name__ == "__main__":
    SAVE_MODEL_PATH = './backup/'+args.tag+'/checkpoint.pt'
    test_model(SAVE_MODEL_PATH)