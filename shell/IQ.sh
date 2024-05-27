DATASET_NAME='fashioniq'
EXP_NAME='AMC'
ID='2'
CUDA_ID=0

cd ..
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u train.py \
    --dataset ${DATASET_NAME}  \
    --name dress \
    --sim_weight 1.0 \
    --max_decay_epoch 50 \
    --lr_decay 40 \
    --num_epochs 50 \
    --img_encoder resnet50 \
    --text_encoder LSTM \
    --grid_num 49 \
    --model_dir ./runs/${EXP_NAME}_${ID}/${DATASET_NAME}/dress \
2>&1 | tee ./logger/${EXP_NAME}_${ID}_${DATASET_NAME}_dress.log


# CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u train.py \
#     --dataset ${DATASET_NAME}  \
#     --name shirt \
#     --sim_weight 1.0 \
#     --max_decay_epoch 50 \
#     --lr_decay 40 \
#     --num_epochs 50 \
#     --img_encoder resnet50 \
#     --text_encoder LSTM \
#     --grid_num 49 \
#     --model_dir ./runs/${EXP_NAME}_${ID}/${DATASET_NAME}/shirt \
# 2>&1 | tee ./logger/${EXP_NAME}_${ID}_${DATASET_NAME}_shirt.log

# CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u train.py \
#     --dataset ${DATASET_NAME}  \
#     --name toptee \
#     --sim_weight 1.0 \
#     --max_decay_epoch 50 \
#     --lr_decay 40 \
#     --num_epochs 50 \
#     --img_encoder resnet50 \
#     --text_encoder LSTM \
#     --grid_num 49 \
#     --model_dir ./runs/${EXP_NAME}_${ID}/${DATASET_NAME}/toptee \
# 2>&1 | tee ./logger/${EXP_NAME}_${ID}_${DATASET_NAME}_toptee.log
