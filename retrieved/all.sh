export CUDA_VISIBLE_DEVICES=1
experiment_name="attention_noise"
# 训练模型
# python train.py --experiment_name $experiment_name
# 预测目前最好效果
python train.py --load_path "/home/user/cdq/FactKG/retrieved/model/$experiment_name.pt" --wandb_use False
# w/o gumbel
# python train.py --load_path "/home/user/cdq/FactKG/retrieved/modelablation_study.pt" --wandb_use False
# w/o gumbel+double
# python train.py --load_path "/home/user/cdq/FactKG/retrieved/model/ablation_study_double.pt" --wandb_use False

# cd /home/user/cdq/FactKG/with_evidence/classifier
# python baseline.py --data_path ../../data/factkg --kg_path ../../data/dbpedia_2015_undirected_light.pickle --mode train