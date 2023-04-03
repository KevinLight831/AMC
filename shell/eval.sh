cd ..
CUDA_ID=0
MODEL_ID=0

DATASET_NAME='fashioniq'
TEST_SET='dress'
EXP_NAME='AMC_sim'
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u eval.py \
    --dataset ${DATASET_NAME}  \
    --name ${TEST_SET} \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir ./runs/${EXP_NAME}_${MODEL_ID}/${DATASET_NAME}/${TEST_SET} \
2>&1 | tee logger/${DATASET_NAME}_${TEST_SET}_${EXP_NAME}_${MODEL_ID}.log

DATASET_NAME='fashioniq'
TEST_SET='toptee'
EXP_NAME='AMC_sim'
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u eval.py \
    --dataset ${DATASET_NAME}  \
    --name ${TEST_SET} \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir ./runs/${EXP_NAME}_${MODEL_ID}/${DATASET_NAME}/${TEST_SET} \
2>&1 | tee logger/${DATASET_NAME}_${TEST_SET}_${EXP_NAME}_${MODEL_ID}.log

DATASET_NAME='fashioniq'
TEST_SET='shirt'
EXP_NAME='AMC_sim'
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u eval.py \
    --dataset ${DATASET_NAME}  \
    --name ${TEST_SET} \
    --data_path /opt/data/private/kevin/data/fashion-iq/ \
    --model_dir ./runs/${EXP_NAME}_${MODEL_ID}/${DATASET_NAME}/${TEST_SET} \
2>&1 | tee logger/${DATASET_NAME}_${TEST_SET}_${EXP_NAME}_${MODEL_ID}.log


DATASET_NAME='shoes'
TEST_SET='shoes'
EXP_NAME='AMC_sim'
CUDA_VISIBLE_DEVICES=${CUDA_ID} python -u eval.py \
    --dataset ${DATASET_NAME}  \
    --name ${TEST_SET} \
    --data_path /opt/data/private/kevin/data/shoes/ \
    --model_dir ./runs/${EXP_NAME}_${MODEL_ID}/${DATASET_NAME} \
2>&1 | tee logger/${DATASET_NAME}_${TEST_SET}_${EXP_NAME}_${MODEL_ID}.log

