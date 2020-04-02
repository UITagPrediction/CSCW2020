import os
import torch
import model
from loader import image_generator
from loader import tag_generator
from function import preprocess,loadGloveModel,loadCategory,get_pre_tag_id,write_json
#
import shutil

THRESHOLD = 0.8

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_types = ['mobile','website','form','chart','grid','list_','dashboard','profile','checkout','landing', 'weather','sport','game','finance','travel','food','ecommerce','music','pink','black','white','green','blue','red','yellow']
model_types = ['sport']

# data loader
df = preprocess()
w2v = loadGloveModel()
test_loader = image_generator()

model_image, _ = model.initialize_model("resnet", feature_extract=True, use_pretrained=True)
model_tag = model.CNN()
model = model.MyEnsemble(model_image, model_tag)
model = model.to(device)

def test_model(tag):
    if device==torch.device('cpu'):
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt',map_location='cpu'))
    else:
        model.load_state_dict(torch.load('backup/'+tag+'/checkpoint.pt'))
    model.eval()

    id_ratio = {}
    for img_inputs, _, paths in test_loader:
        img_inputs = img_inputs.to(device)
        tag_inputs = tag_generator(paths, df, w2v, tag).to(device)
        outputs = torch.sigmoid(model(img_inputs, tag_inputs))
        # {id: ratio}
        for i in range(len(outputs)):
            id_ratio[int(paths[i].split('/')[-1].split('.')[0])] = outputs[i].tolist()[1]
            # if outputs[i].tolist()[1]>0.5:
            #     shutil.copyfile(paths[i],'./saves/y/'+paths[i].split('/')[-1])
            # else:
            #     shutil.copyfile(paths[i],'./saves/n/'+paths[i].split('/')[-1])
    return id_ratio

if __name__ == "__main__":
    # inital
    return_result = {}
    for id in df.id:
        return_result[id] = []
    # predict each model
    for t in model_types:
        print('-'*10)
        print('Processing '+ t+' .....')
        # predict    
        id_ratio = test_model(t)
        if t == 'list_': t = 'list'
        # pre tag design process: {id: [(tag,ratio), ... ]}
        pre_tag_id = get_pre_tag_id(df,t)
        # for id in pre_tag_id:
        #     return_result[id].append((t,id_ratio[id]))
        # new tag design process: 1. filter threshld 2. {id: [(tag,ratio), ... ]}
        for id in df.id:
            if id in pre_tag_id:
                return_result[id].append((t,id_ratio[id]))
            else:
                if id_ratio[id] >= THRESHOLD:
                    return_result[id].append((t,id_ratio[id]))
    # write to json file
    write_json(df,return_result)