import torch
import glob
import sys
# import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
# from skimage.metrics import peak_signal_noise_ratio
from hidden.models import HiddenEncoder, HiddenDecoder, EncoderWithJND, EncoderDecoder
from hidden.attenuations import JND
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

class Params():
    def __init__(self, encoder_depth:int, encoder_channels:int, decoder_depth:int, decoder_channels:int, num_bits:int,
                attenuation:str, scale_channels:bool, scaling_i:float, scaling_w:float):
        # encoder and decoder parameters
        self.encoder_depth = encoder_depth
        self.encoder_channels = encoder_channels
        self.decoder_depth = decoder_depth
        self.decoder_channels = decoder_channels
        self.num_bits = num_bits
        # attenuation parameters
        self.attenuation = attenuation
        self.scale_channels = scale_channels
        self.scaling_i = scaling_i
        self.scaling_w = scaling_w

NORMALIZE_IMAGENET = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
UNNORMALIZE_IMAGENET = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
default_transform = transforms.Compose([transforms.ToTensor(), NORMALIZE_IMAGENET])

params = Params(
    encoder_depth=4, encoder_channels=64, decoder_depth=8, decoder_channels=64, num_bits=48,
    attenuation="jnd", scale_channels=False, scaling_i=1, scaling_w=1.5
)

decoder = HiddenDecoder(
    num_blocks=params.decoder_depth, 
    num_bits=params.num_bits, 
    channels=params.decoder_channels
)
encoder = HiddenEncoder(
    num_blocks=params.encoder_depth, 
    num_bits=params.num_bits, 
    channels=params.encoder_channels
)
attenuation = JND(preprocess=UNNORMALIZE_IMAGENET) if params.attenuation == "jnd" else None
encoder_with_jnd = EncoderWithJND(
    encoder, attenuation, params.scale_channels, params.scaling_i, params.scaling_w
)

ckpt_path = "./hidden/ckpts/hidden_replicate.pth"

state_dict = torch.load(ckpt_path, map_location='cpu')['encoder_decoder']
encoder_decoder_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
encoder_state_dict = {k.replace('encoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'encoder' in k}
decoder_state_dict = {k.replace('decoder.', ''): v for k, v in encoder_decoder_state_dict.items() if 'decoder' in k}

encoder.load_state_dict(encoder_state_dict)
decoder.load_state_dict(decoder_state_dict)

encoder_with_jnd = encoder_with_jnd.to(device).eval()
decoder = decoder.to(device).eval()

if __name__ == "__main__":
    # Get the first argument
    first_arg = sys.argv[1]
    second_arg = sys.argv[2]

    print(first_arg)

    # Define the directory name
    source_dir_name = first_arg
    target_dir_name = second_arg

    target_dir = Path(target_dir_name)
    target_dir.mkdir(exist_ok=True)

    # Get a list of all image files in the directory
    image_files = glob.glob(f"{source_dir_name}/*.[jp][pn]g")

    # Now image_files is a list of filenames
    for filename in image_files:
        print(filename)

        # load image
        # img = Image.open("assets/better.png").convert('RGB')
        img = Image.open(filename).convert('RGB')
        # img = img.resize((512, 512), Image.BICUBIC)
        img_pt = default_transform(img).unsqueeze(0).to(device)

        print(img_pt.shape)

        # create message
        random_msg = False
        if random_msg:
            msg_ori = torch.randint(0, 2, (1, params.num_bits), device=device).bool() # b k
        else:
            msg_ori = torch.Tensor(str2msg("111010110101000001010111010011010100010000100111")).unsqueeze(0)
        msg = 2 * msg_ori.type(torch.float) - 1 # b k

        msg = msg.cuda()
        msg_ori = msg_ori.cuda()

        # encode
        img_w = encoder_with_jnd(img_pt, msg)
        clip_img = torch.clamp(UNNORMALIZE_IMAGENET(img_w), 0, 1)
        clip_img = torch.round(255 * clip_img) / 255
        clip_img = transforms.ToPILImage()(clip_img.squeeze(0).cpu())
        
        filename = Path(filename)
        # filename = filename.with_suffix('.jpg')
        filename = filename.with_suffix('.png')
        clip_img.save(target_dir / filename.name)

        # decode
        ft = decoder(default_transform(clip_img).unsqueeze(0).to(device))
        decoded_msg = ft > 0 # b k -> b k
        accs = (~torch.logical_xor(decoded_msg, msg_ori)) # b k -> b k
        print(f"Message: {msg2str(msg_ori.squeeze(0).cpu().numpy())}")
        print(f"Decoded: {msg2str(decoded_msg.squeeze(0).cpu().numpy())}")
        print(f"Bit Accuracy: {accs.sum().item() / params.num_bits}")
