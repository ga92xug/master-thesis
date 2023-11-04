from omegaconf import DictConfig
import torch
from typing import List, Tuple
import torch
import torch.nn as nn
#import foolbox as fb
#from torchattacks import PGD
#from robustbench.utils import clean_accuracy
import time
import sys
import os


sys.path.append(f"{os.getcwd()}")
from experiments.d_application_experiment._4_adversarial_attack.attack_options.autoattack_option import run_autoattack
from experiments.d_application_experiment._4_adversarial_attack.attack_options.foolbox_option import foolbox_attack
from training.logger import Custom_Logger
from training import utils
from training.model_instantiate import hydra_compose 


def adversarial_attack(
        mode: str,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        cfg: DictConfig,
        device: torch.device,
        logger: Custom_Logger = None,
    ): 
    if logger is not None:
        message = f"-"*50 + "\nAdversarial Attack"
        logger.print_verbose_check(0, message)

    mean = torch.tensor(cfg.training.dataset.channel_wise_mean_images).to(device)
    std = torch.tensor(cfg.training.dataset.channel_wise_std_images).to(device)

    # get results
    if mode == "AutoAttack":
        results = run_autoattack(
            model=model,
            dataloader=dataloader,
            device=device,
            mean=mean,
            std=std,
            verbose=cfg.other.verbose,
            num_classes=cfg.training.dataset.n_out_classes,
        )
    elif mode == "Foolbox":
        results = foolbox_attack(
            model=model,
            device=device,
            dataloader=dataloader,
            mean=mean,
            std=std,
        )
    else:
        raise NotImplementedError(f"mode {mode} not implemented")

    # report results
    if logger is not None:
        logger.log(results, "adversarial_attack", 1)
    else:
        print(results)

    return results


def debug_adversarial_attack():
    """
    This function can be used to debug the auto_attack_eval function.
    """

    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model, dataloaders, cfg = hydra_compose(overrides=["training=DeepDRiD-training", "model=eq_nasnet"])

    # load model
    run_id = "myw54oac"
    path = f"/home/frischs/outputs/{run_id}/model.pth"
    model.load_state_dict(torch.load(path))

    adversarial_attack(
        mode="Foolbox",
        model=model,
        dataloader=dataloaders["test"],
        cfg=cfg,
        device=device,
    )



if __name__ == "__main__":
    debug_adversarial_attack()



"""
mean= [0.38250651955604553, 0.23592887818813324, 0.14257268607616425]
std= [0.324453204870224, 0.21356910467147827, 0.15683040022850037]
mean = torch.tensor(mean).to(device)
std = torch.tensor(std).to(device)

train_dataloader = dataloaders["train"]
for i, out_dataloader in enumerate(train_dataloader):
    images, labels, meta_data = utils.get_out_dataloader(out_dataloader)
    print(images.shape)
    print(labels.shape)
    break

images = images.to(device)
labels = labels.to(device)

images = images * std[:, None, None] + mean[:, None, None]







adversary = configure_attack()

# adversary = AutoAttack(model, norm='Linf', eps=8/255, log_path=aa_state_path)
x_adv = adversary.run_standard_evaluation(images, labels)
#to_print = adversary.clean_accuracy(x_adv, labels)
#print(f"Clean accuracy: {to_print}")

with open(aa_state_path, 'r') as f:
    content = f.read()
    result = extract_percentages(content)

print("Read from file:")
print(result)

quit()
# aa_state = EvaluationState.from_disk(aa_state_path)
#assert aa_state.robust_flags is not None
adv_accuracy = aa_state.robust_flags.mean().item()


#adv_accuracy = clean_accuracy(model, x_adv, labels)
quit()

x_adv


adversary = AutoAttack(model, norm='Linf', eps=8/255)


#model = model.eval()

mean= [0.38250651955604553, 0.23592887818813324, 0.14257268607616425]
std= [0.324453204870224, 0.21356910467147827, 0.15683040022850037]

mean = torch.tensor(mean).to(device)
std = torch.tensor(std).to(device)

preprocessing = dict(mean=mean, std=std, axis=-3)
bounds = (0, 1)
fmodel = fb.PyTorchModel(model, bounds=bounds, preprocessing=preprocessing)

 [markdown]
# ## Load model and data


images.min().item()


fb.utils.accuracy(fmodel, images, labels)

attack = fb.attacks.LinfPGD()
epsilons = [0.01] # [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]
_, advs, success = attack(fmodel, images, labels, epsilons=epsilons)


success


fmodel = PyTorchModel(model, bounds = (0, 1), preprocessing = dict(mean = mean, std = std)) 
epsilons = [0.01, 0.03, 0.1, 0.3, 0.5]
cnt, total = torch.zeros(len(epsilons)).to(device),\
                torch.zeros(len(epsilons)).to(device)



correct = torch.zeros(len(epsilons)).to(device)

for _, (images, labels) in enumerate(eval_loader):
    images = images.to(device)
    labels = labels.to(device)

    images = images * std[:, None, None] + mean[:, None, None]

    _, advs_list, success = attack(fmodel, images, labels, epsilons = epsilons)
    cnt += success.sum(axis = 1)
    total += images.shape[0]

    for i, advs in enumerate(advs_list):
        preds = model(advs).argmax(dim=1)
        correct[i] += (preds == labels).sum().item()  # Compute accuracy for each epsilon

print(f"Success rate vector: {cnt / total}")
print(f"Accuracy vector for each epsilon: {correct / total}")


#sys.path.insert(0, '..')
import robustbench
from robustbench.data import load_cifar10
from robustbench.utils import load_model, clean_accuracy

#images, labels = load_cifar10(n_examples=5)
print('[Data loaded]')

device = "cuda"
#model = load_model('Standard', norm='Linf').to(device)
acc = clean_accuracy(model, images.to(device), labels.to(device))
print('[Model loaded]')
print('Acc: %2.2f %%'%(acc*100))

 [markdown]
# ## Adversarial Attack


#from utils import imshow, get_pred
from autoattack import AutoAttack
adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
adversary.apgd.n_restarts = 1
x_adv = adversary.run_standard_evaluation(x_test, y_test)


atk = PGD(model, eps=8/255, alpha=2/225, steps=10, random_start=True)
print(atk)


import robustbench
from robustbench import benchmark


from robustbench import benchmark
clean_acc, robust_acc = benchmark(model,
                                  dataset='cifar10',
                                  threat_model='Linf')


# When normalization used:
atk.set_normalization_used(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


adv_images = atk(images, labels)


output = model(adv_images)
pred = output.argmax(dim=1, keepdim=True)
pred


idx = 0
pre = get_pred(model, adv_images[idx:idx+1], device)
imshow(adv_images[idx:idx+1], title="True:%d, Pre:%d"%(labels[idx], pre))

"""
