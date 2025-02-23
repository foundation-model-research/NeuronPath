from pathlib import Path
import jsonlines
import numpy as np
import os

def load_results_jsonl(path):
    with jsonlines.open(path) as reader:
        remove_accuracy = []
        base_accuracy = []
        enhance_accuracy  =  []
        prune_accuracy=[]
        remove_pro = []
        enhance_pro = []
        _data = [obj for obj in reader]

        for _ in _data:
            for one_data in _:
                # remove_accuracy.append(one_data['remove'])
                base_accuracy.append(one_data['base'])
                # enhance_accuracy.append(one_data['enhance'])
                prune_accuracy.append(one_data['prune'])
        
        # remove_accuracy = np.array(remove_accuracy).squeeze(1)
        base_accuracy = np.array(base_accuracy).squeeze(1)
        # enhance_accuracy = np.array(enhance_accuracy).squeeze(1)
        prune_accuracy = np.array(prune_accuracy).squeeze(1)

        # remove_pro = remove_accuracy/base_accuracy
        # enhance_pro = enhance_accuracy/base_accuracy
        prune_pro = prune_accuracy/base_accuracy

    # return (remove_accuracy, base_accuracy, enhance_accuracy, prune_accuracy), (remove_pro, enhance_pro, prune_pro)
    return (base_accuracy, prune_accuracy), prune_pro



def getResult(JA_PATH):
    print(JA_PATH)
    # INFLUENCE_PATTERN_PATH = './edit_ip_ViT_B_16'
    # BASE_PATH = './edit_base_ViT_B_16'

    base_accuracyList = []
    prune_accuracyList = []
    ja_remove_lst = []
    ja_enhance_lst = []
    ja_prune_lst = []
    base_remove_lst = []
    base_enhance_lst = []

    for class_idx in range(1000):
        # influence_pattern_path = os.path.join(INFLUENCE_PATTERN_PATH, f"imgnet_all-{int(class_idx)}.rlt.jsonl")
        ja_path = os.path.join(JA_PATH, f"imgnet_all-{int(class_idx)}.rlt.jsonl")

        # base_path = os.path.join(BASE_PATH, f"imgnet_all-{int(class_idx)}.rlt.jsonl")

        # _, (ip_remove, ip_enhance) = load_results_jsonl(influence_pattern_path)
        (base_accuracy, prune_accuracy), ja_prune = load_results_jsonl(ja_path)
        
        base_accuracy=base_accuracy[:50]
        prune_accuracy=prune_accuracy[:50]
        
        base_accuracyList.append(base_accuracy)
        prune_accuracyList.append(prune_accuracy)
        
        ja_prune=ja_prune[:12]
        ja_prune_lst.append(ja_prune)
        
        # base_remove_lst.append(base_remove)
        # base_enhance_lst.append(base_enhance)

    mean_ja_prune_vit_b_16 = np.array(ja_prune_lst).mean(1)
    mean_base_accuracy = np.array(base_accuracyList).mean(1)
    mean_prune_accuracy = np.array(prune_accuracyList).mean(1)
    

    prune_vit_b_16 = [
        mean_base_accuracy,
        mean_prune_accuracy,
        mean_ja_prune_vit_b_16
    ]
    
    return prune_vit_b_16
    

baseline_vit_b_16 = getResult('/path/to/your/data')
np.savez('data/baseline_vit_b_16', *baseline_vit_b_16)


folder='Ablation' 

dataFolder=Path(f'/path/to/your/data/{folder}')
for experimentFolder in dataFolder.iterdir():
    if experimentFolder.is_dir():
        if Path(f'data/{folder}/{experimentFolder.name}.npz').exists():
            continue
        np.savez(f'data/{folder}/{experimentFolder.name}.npz', *getResult(str(experimentFolder)))