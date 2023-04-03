DATASET_NAME='fashioniq'
EXP_NAME='AMC_sim'
CUDA_ID=0

cd ..
CUDA_VISIBLE_DEVICES=${CUDA_ID} nohup python -u eval_ensemble.py \
    --dataset ${DATASET_NAME}  \
    --name dress \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir1 ./runs/${EXP_NAME}_0/${DATASET_NAME}/dress \
    --model_dir2 ./runs/${EXP_NAME}_1/${DATASET_NAME}/dress \
>'logger/eval_ensmeble_'${EXP_NAME}'.log' 2>&1

CUDA_VISIBLE_DEVICES=${CUDA_ID} nohup python -u eval_ensemble.py \
    --dataset ${DATASET_NAME}  \
    --name shirt \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir1 ./runs/${EXP_NAME}_0/${DATASET_NAME}/shirt \
    --model_dir2 ./runs/${EXP_NAME}_1/${DATASET_NAME}/shirt \
>>'logger/eval_ensmeble_'${EXP_NAME}'.log' 2>&1

CUDA_VISIBLE_DEVICES=${CUDA_ID} nohup python -u eval_ensemble.py \
    --dataset ${DATASET_NAME}  \
    --name toptee \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir1 ./runs/${EXP_NAME}_0/${DATASET_NAME}/toptee \
    --model_dir2 ./runs/${EXP_NAME}_1/${DATASET_NAME}/toptee \
>>'logger/eval_ensmeble_'${EXP_NAME}'.log' 2>&1

DATASET_NAME='shoes'
CUDA_VISIBLE_DEVICES=${CUDA_ID} nohup python -u eval_ensemble.py \
    --dataset ${DATASET_NAME}  \
    --name shoes \
    --data_path /opt/data/private/kevin/data/shoes/ \
    --model_dir1 ./runs/${EXP_NAME}_0/${DATASET_NAME} \
    --model_dir2 ./runs/${EXP_NAME}_1/${DATASET_NAME} \
>>'logger/eval_ensmeble_'${EXP_NAME}'.log' 2>&1