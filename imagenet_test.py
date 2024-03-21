import torch
import torchvision.models as tm
import os
import csv

os.environ['CUDA_VISIBLE_DEVICES']='1'

from imagenet import get_imagenet_dataloader

if __name__ == '__main__':
    model = tm.resnet34(pretrained=True)
    model = model.cuda()
    model.eval()

    train_loader, test_loader = get_imagenet_dataloader(batch_size=100)

    for idx, (input, target) in enumerate(test_loader):

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        output_label = torch.argmax(output, dim=1)
        # with open('bird_80.txt','a+') as f:
        #     csv_writer=csv.writer(f)
        #     csv_writer.writerow([output_label.detach().cpu().numpy()])
        print(output_label)
