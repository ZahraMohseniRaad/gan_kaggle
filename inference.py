import argparse
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from generator import getGenerator
from dataset import testloader
from torchvision.utils import save_image

def load_model(device):
    generator = getGenerator()
    return generator


def generate_image(generator, x, device):
    with torch.no_grad():
        x = x.to(device)
        output_tensor = generator(x)
    return output_tensor

def main():
    parser = argparse.ArgumentParser(description='Generate an output image from an input image using a pre-trained Pix2Pix model.')
    parser.add_argument('input_image_path', type=str, help='Path to the input image')
    parser.add_argument('output_image_path', type=str, help='Path to save the output image')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (default: cpu)')

    args = parser.parse_args()
    
    # dataset
    loader = testloader()
    # generator
    gen,_,_ = load_model(args.device)

    x, y = next(iter(loader))
    x = x.to(args.device)
    y = y.to(args.device)
    output = generate_image(gen, x, args.device)
    final_output = torch.cat([x*.5+.5, output*.5+.5, y*.5+.5], 0)
    save_image(final_output, "/content/fake_output.png")

    plt.imshow(final_output)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
