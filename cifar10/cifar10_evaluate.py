import subprocess
import pandas as pd


###
dataset = 'cifar10' ##
seeds = [0, 1, 2]
sparsities = [0, 20, 40, 60] #[0, 20, 40, 60]##
noise_rates = [0, 10, 20] #[0, 10, 20]##
methods = ['kmeans', 'no_clean', 'cl'] #['kmeans', 'kmeans_pred', 'kmeans_pred_dyn', 'no_clean', 'cl']
model_arch = 'resnet50'
###


evaluate_template = '\
python {dataset}_train.py --evaluate \
--arch {model_arch} --gpu 0 --workers 4 --batch-size 256 \
--resume {model_path} \
datasets/datasets/{dataset}/{dataset}/ \
'
folder_template = '{dataset}_noisy_labels__frac_zero_noise_rates__0_{sparsity}__noise_amount__0_{noise}/seed_{seed}'
model_path_template = '{dataset}_training_masked/{base_folder}/{method}/model_{model_arch}__masked_{ckpt}.pth.tar'
train_model_path_template = '{dataset}_training/{base_folder}/model_{model_arch}_{ckpt}.pth.tar'

results = []
for seed in seeds:
    for sparsity in sparsities:
        for noise in noise_rates:
            if noise < 1e-2 and sparsity>0:
                continue
            sparsity_str = '{0:0=2d}'.format(sparsity)
            noise_str = '{0:0=2d}'.format(noise)
            base_folder = folder_template.format(
                dataset=dataset, sparsity=sparsity_str, noise=noise_str, seed=seed)
            
            for method in methods:
                for ckpt in ['best', '_checkpoint']:
                    model_path = model_path_template.format(
                        dataset=dataset, base_folder=base_folder, method=method, model_arch=model_arch, ckpt=ckpt)

                    cmd = evaluate_template.format(
                        dataset=dataset, model_arch=model_arch, model_path=model_path)
                    result = subprocess.check_output(cmd, shell=True)
                    acc1, _,acc5, _,acc_avg, _,acc_min = result.split(b"* Acc@1 ")[-1].strip().split()
                    acc1, acc5, acc_avg, acc_min = float(acc1), float(acc5), float(acc_avg), float(acc_min)
                    
                    results.append({
                        'method': method,
                        'test_ckpt': 'best' if ckpt=='best' else 'last',
                        'seed': seed,
                        'frac_zero_noise_rates': sparsity/100.,
                        'noise_amount': noise/100.,
                        'acc1': acc1,
                        'acc5': acc5,
                        'acc_avg': acc_avg,
                        'acc_min': acc_min,
                    })

df = pd.DataFrame(results)
df.to_csv('evaluation.csv', index=False)
