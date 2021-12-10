import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
from torchvision.transforms import ToTensor
import models.crnn as crnn

model_path = './crnn.pth'
img_path = './data/demo.png'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# model_path = '/home/ngoc/work/ocr/crnn.pytorch/netCRNN_100.pth'
# img_path = '/home/ngoc/work/ocr/crnn-license-plate-OCR/images/photo_2021-11-29_14-45-22.jpg'
# img_path = '/home/ngoc/work/ocr/crnn.pytorch/valset/11A-652.75.jpg'
# img_path = '/home/ngoc/work/ocr/crnn.pytorch/images/56FCYikaub.jpg'
# alphabet = '-.0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ncl = len(alphabet)+1 #39

model = crnn.CRNN(32, 1, ncl, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')), strict=False)
model.eval()

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
image = Image.open(img_path).convert('L')
image = transformer(image)
if torch.cuda.is_available():
    image = image.cuda()
image = image.view(1, *image.size())
image = Variable(image)

preds = model(image)
print("\ntime step 1:\n",preds[0])
print("\ntime step 3:\n",preds[2])
aa, preds = preds.max(2) # return max of value and it's index

preds = preds.transpose(1, 0).contiguous().view(-1)
print(aa[0], preds)
preds_size = Variable(torch.IntTensor([preds.size(0)]))
raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
print('%-20s => %-20s' % (raw_pred, sim_pred))
