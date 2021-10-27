from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18 as Net
import os, torch, argparse
from utils.image_folder import ImageFolder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict(dataset, weights, normalize=(0.4612, 0.1470)):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(normalize[0],normalize[1])
    ])
    test_datasets = ImageFolder(dataset, transform=transform)
    batch_size = len(test_datasets)

    test_loader = DataLoader(test_datasets, batch_size=batch_size, shuffle=True)

    state_dict = torch.load(weights, map_location=device)
    model = Net(**{'num_classes': 2})


    model.load_state_dict(state_dict['cls'])
    model.eval()
    model.to(device)

    test_iter = iter(test_loader)
    images, lables, paths = test_iter.next()
    images, lables = images.to(device), lables.to(device)

    with torch.no_grad():

        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        true_sex_list = lables.cpu().numpy().tolist()
        pre_sex_list = predicted.cpu().numpy().tolist()
        person_list = []
        acc = 0
        male_acc = 0
        female_acc = 0
        for i in range(batch_size):
            if lables[i] == predicted[i]:
                acc += 1
                if true_sex_list[i] == 1:
                    male_acc += 1
                else:
                    female_acc += 1
            data = {
                'true_sex': true_sex_list[i],
                'pre_sex': pre_sex_list[i]
            }
            if paths[i].split('_')[3] == 'N':
                data['age'] = 27.00
            else:
                data['age'] = float(paths[i].split('_')[3])
            person_list.append(data)
        age_acc_list = [{'true_male':0,'pre_male':0,'true_female':0,'pre_female':0},
                        {'true_male':0,'pre_male':0,'true_female':0,'pre_female':0},
                        {'true_male':0,'pre_male':0,'true_female':0,'pre_female':0}]
        for person in person_list:
            if person['age'] < 27 and person['age'] >= 18:
                if person['true_sex']== 1:
                    age_acc_list[0]['true_male'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[0]['pre_male'] += 1
                else:
                    age_acc_list[0]['true_female'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[0]['pre_female'] += 1
            elif 27 <= person['age'] and person['age'] < 47:
                if person['true_sex']== 1:
                    age_acc_list[1]['true_male'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[1]['pre_male'] += 1
                else:
                    age_acc_list[1]['true_female'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[1]['pre_female'] += 1
            elif person['age'] >= 47:
                if person['true_sex']== 1:
                    age_acc_list[2]['true_male'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[2]['pre_male'] += 1
                else:
                    age_acc_list[2]['true_female'] += 1
                    if person['true_sex'] == person['pre_sex']:
                        age_acc_list[2]['pre_female'] += 1
        print("{} items, {} correct, accuracy: {}".format(batch_size, acc, acc/batch_size))
        print()
        print("{} male, {} correct, accuracy: {}".format(true_sex_list.count(1), male_acc, male_acc/true_sex_list.count(1) if true_sex_list.count(1) != 0 else 0))
        print("{} female, {} correct, accuracy: {}".format(true_sex_list.count(0), female_acc, female_acc/true_sex_list.count(0) if true_sex_list.count(0) != 0 else 0))
        return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/predict.pkl', help='model.pkl path(s)')
    parser.add_argument('--dataset', type=str, default='runs/test/crop', help='file/dir/URL/glob, 0 for webcam')
    opt = parser.parse_args()
    print(opt)
    predict(**vars(opt))
