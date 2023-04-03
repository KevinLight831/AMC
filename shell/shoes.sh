DATASET_NAME='shoes'
EXP_NAME='AMC'
ID='0'
CUDA_ID=0

cd ..

CUDA_VISIBLE_DEVICES=${CUDA_ID} nohup python -u train.py \
    --dataset ${DATASET_NAME}  \
    --name shoes \
    --sim_weight 1.0 \
    --max_decay_epoch 50 \
    --lr_decay 40 \
    --num_epochs 50 \
    --img_encoder resnet50 \
    --text_encoder LSTM \
    --grid_num 49 \
    --model_dir ./runs/${EXP_NAME}_${ID}/${DATASET_NAME}/ \
>'logger/'${EXP_NAME}'_'${ID}'_'${DATASET_NAME}'.log' 2>&1

CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u train.py \
    --dataset ${DATASET_NAME}  \
    --name toptee \
    --sim_weight 1.0 \
    --max_decay_epoch 50 \
    --lr_decay 40 \
    --num_epochs 50 \
    --img_encoder resnet50 \
    --text_encoder LSTM \
    --grid_num 49 \
    --model_dir ./runs/${EXP_NAME}_${ID}/${DATASET_NAME}/ \
2>&1 | tee ./logger/${EXP_NAME}_${ID}_${DATASET_NAME}.log